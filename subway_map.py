"""
Live subway map drawn with matplotlib.

Draws the subway lines (from the MTA's static schedule) over land and water.
Each train is plotted between the station it last left and the station it is
heading to, sliding toward that station using the MTA's arrival predictions.
The feeds refresh every 30 seconds; the trains move every second.

Tick or untick the lines on the right to show or hide them.

    python subway_map.py            # live window
    python subway_map.py --once     # save one snapshot to live_map.png

Files downloaded on the first run (kept next to this script):
  gtfs_subway.zip   MTA subway schedule, about 6 MB
  ne_10m_land.zip   Natural Earth land outlines, about 3 MB (for the water contrast)

Note: train positions between stations are estimates from the predicted
arrival times, not GPS. The feeds don't say how many passengers are aboard.
"""

import argparse
import csv
import io
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shapefile
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, CheckButtons
from google.transit import gtfs_realtime_pb2

HERE = Path(__file__).resolve().parent
GTFS_ZIP = HERE / "gtfs_subway.zip"
GTFS_URL = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip"
LAND_ZIP = HERE / "ne_10m_land.zip"
LAND_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
FEED_BASE = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/"
FEEDS = ["nyct%2Fgtfs", "nyct%2Fgtfs-ace", "nyct%2Fgtfs-bdfm", "nyct%2Fgtfs-g",
         "nyct%2Fgtfs-jz", "nyct%2Fgtfs-nqrw", "nyct%2Fgtfs-l", "nyct%2Fgtfs-si"]
FEED_REFRESH_SECONDS = 30    # how often to fetch the MTA feeds
FRAME_MS = 1000              # how often the trains move
STOPPED_AT = 1               # GTFS-realtime VehiclePosition.stopped_at

WATER = "#0b2a4a"
LAND = "#1f1f1f"

ROUTE_COLORS = {
    "1": "#EE352E", "2": "#EE352E", "3": "#EE352E",
    "4": "#00933C", "5": "#00933C", "6": "#00933C",
    "7": "#B933AD", "7X": "#B933AD",
    "A": "#0039A6", "C": "#0039A6", "E": "#0039A6", "SI": "#0039A6",
    "B": "#FF6319", "D": "#FF6319", "F": "#FF6319", "M": "#FF6319",
    "G": "#6CBE45", "J": "#996633", "Z": "#996633", "L": "#A7A9AC",
    "N": "#FCCC0A", "Q": "#FCCC0A", "R": "#FCCC0A", "W": "#FCCC0A",
    "S": "#808183", "FS": "#808183", "GS": "#808183", "H": "#808183",
}


def download_if_missing(path, url, label):
    if not path.exists():
        print(f"Downloading {label} (one time)…")
        urllib.request.urlretrieve(url, path)
    return path


def read_csv(z, name):
    return csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8"))


def load_static():
    """Return (stops, lines, trip_stops).
    stops: stop_id -> (lat, lon)
    lines: list of (route, [(lon, lat), ...])
    trip_stops: trip_id -> ordered list of stop_ids the trip visits."""
    z = zipfile.ZipFile(download_if_missing(GTFS_ZIP, GTFS_URL, "MTA subway schedule"))
    stops = {}
    for r in read_csv(z, "stops.txt"):
        stops[r["stop_id"]] = (float(r["stop_lat"]), float(r["stop_lon"]))

    shape_route = {}
    for r in read_csv(z, "trips.txt"):
        shape_route.setdefault(r["shape_id"], r["route_id"])

    points = defaultdict(list)
    for r in read_csv(z, "shapes.txt"):
        points[r["shape_id"]].append((int(r["shape_pt_sequence"]), float(r["shape_pt_lon"]), float(r["shape_pt_lat"])))
    lines = []
    for sid, pts in points.items():
        pts.sort()
        lines.append((shape_route.get(sid, "?"), [(lon, lat) for _, lon, lat in pts]))

    visits = defaultdict(list)
    for r in read_csv(z, "stop_times.txt"):
        visits[r["trip_id"]].append((int(r["stop_sequence"]), r["stop_id"]))
    trip_stops = {trip: [stop for _, stop in sorted(v)] for trip, v in visits.items()}
    return stops, lines, trip_stops


def load_land(bbox):
    """Land polygons (lon, lat) that overlap bbox = (min_lon, min_lat, max_lon, max_lat)."""
    z = zipfile.ZipFile(download_if_missing(LAND_ZIP, LAND_URL, "land outlines"))
    reader = shapefile.Reader(shp=io.BytesIO(z.read("ne_10m_land.shp")),
                              dbf=io.BytesIO(z.read("ne_10m_land.dbf")))
    min_lon, min_lat, max_lon, max_lat = bbox
    polys = []
    for shape in reader.shapes():
        x0, y0, x1, y1 = shape.bbox
        if x1 < min_lon or x0 > max_lon or y1 < min_lat or y0 > max_lat:
            continue
        parts = list(shape.parts) + [len(shape.points)]
        for a, b in zip(parts[:-1], parts[1:]):
            polys.append(shape.points[a:b])
    return polys


def locate(stop_id, stops):
    """Feeds report platforms like 'N10S'; fall back to the station when the platform isn't listed."""
    if stop_id in stops:
        return stops[stop_id]
    return stops.get(stop_id[:-1])


def fetch_feed(name):
    req = urllib.request.Request(FEED_BASE + name, headers={"User-Agent": "subway-map/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        msg = gtfs_realtime_pb2.FeedMessage()
        msg.ParseFromString(resp.read())
    return msg


def fetch_model(stops, trip_stops):
    """Read the feeds and work out, for each train, the station it is leaving,
    the station it is heading to, and when it is due there."""
    now = time.time()
    predictions = defaultdict(dict)   # trip_id -> {stop_id: arrival timestamp}
    vehicles = []                     # (route, trip_id, stop_id, status)
    with ThreadPoolExecutor(max_workers=len(FEEDS)) as pool:
        futures = [pool.submit(fetch_feed, name) for name in FEEDS]
        for fut in futures:
            try:
                feed = fut.result()
            except Exception as exc:
                print("feed error:", exc)
                continue
            for entity in feed.entity:
                if entity.HasField("trip_update"):
                    tu = entity.trip_update
                    for stu in tu.stop_time_update:
                        ts = stu.arrival.time or stu.departure.time
                        if ts:
                            predictions[tu.trip.trip_id][stu.stop_id] = ts
                if entity.HasField("vehicle"):
                    v = entity.vehicle
                    vehicles.append((v.trip.route_id or "?", v.trip.trip_id, v.stop_id, v.current_status))

    models = []
    for route, trip, stop_id, status in vehicles:
        to_spot = locate(stop_id, stops)
        if to_spot is None:
            continue
        frm = None
        seq = trip_stops.get(trip, [])
        if status != STOPPED_AT and stop_id in seq:
            idx = seq.index(stop_id)
            if idx > 0:
                frm = locate(seq[idx - 1], stops)
        t1 = predictions.get(trip, {}).get(stop_id)
        models.append({"route": route, "frm": frm, "to": to_spot, "t0": now, "t1": t1})
    return models, now


def position(model, t):
    """Where a train is at time t: sliding from the last station to the next one."""
    frm, to, t0, t1 = model["frm"], model["to"], model["t0"], model["t1"]
    if frm is None or t1 is None or t1 <= t0:
        return to
    p = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
    return (frm[0] + (to[0] - frm[0]) * p, frm[1] + (to[1] - frm[1]) * p)


def spread_overlaps(points):
    """Trains at the same spot would sit on top of each other: fan them out in a small circle."""
    per_spot = Counter((round(lat, 5), round(lon, 5)) for _, lat, lon in points)
    seen = Counter()
    out = []
    for route, lat, lon in points:
        key = (round(lat, 5), round(lon, 5))
        n, i = per_spot[key], seen[key]
        seen[key] += 1
        if n > 1:
            angle = 2 * np.pi * i / n
            r = 0.0009
            lat += r * np.sin(angle)
            lon += r * np.cos(angle) / np.cos(np.radians(lat))
        out.append((route, lat, lon))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="save one snapshot to live_map.png and exit")
    args = parser.parse_args()

    stops, lines, trip_stops = load_static()
    all_lon = [x for _, c in lines for x, _ in c]
    all_lat = [y for _, c in lines for _, y in c]
    pad = 0.01
    bbox = (min(all_lon) - pad, min(all_lat) - pad, max(all_lon) + pad, max(all_lat) + pad)

    fig = plt.figure(figsize=(11, 11))
    fig.set_facecolor("black")
    ax = fig.add_axes([0.01, 0.01, 0.74, 0.94])
    ax.set_facecolor(WATER)
    ax.set_xticks([])          # hide ticks and frame but keep the background colour
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect(1 / np.cos(np.radians(np.mean(all_lat))))

    # Land sits on top of the water colour, so the coastline and harbors show up.
    land = [Polygon(p, closed=True) for p in load_land(bbox)]
    ax.add_collection(PatchCollection(land, facecolor=LAND, edgecolor="none", zorder=1))

    # One collection per route, so each line can be shown or hidden.
    by_route = defaultdict(list)
    for route, coords in lines:
        if route != "?":
            by_route[route].append(coords)
    route_collections = {}
    for route in sorted(by_route):
        coll = LineCollection(by_route[route], colors=ROUTE_COLORS.get(route, "#777777"),
                              linewidths=1.2, alpha=0.9, zorder=2)
        ax.add_collection(coll)
        route_collections[route] = coll

    station_xy = np.array(list(stops.values()))
    ax.scatter(station_xy[:, 1], station_xy[:, 0], s=2, c="#e6e6e6", zorder=3)
    trains_plot = ax.scatter([], [], s=110, edgecolors="black", linewidths=0.9, zorder=4)
    title = ax.set_title("", color="white", fontsize=13, loc="left")

    # Line picker on the right.
    visible = {route: True for route in route_collections}
    routes = sorted(route_collections)
    box_ax = fig.add_axes([0.77, 0.12, 0.22, 0.82])
    box_ax.set_facecolor("black")
    checks = CheckButtons(box_ax, routes, [True] * len(routes))
    for route, label in zip(routes, checks.labels):
        label.set_color(ROUTE_COLORS.get(route, "#777777") if route not in ("N", "Q", "R", "W") else "#FCCC0A")
        label.set_fontsize(10)
        label.set_fontweight("bold")

    def set_route(route, state):
        visible[route] = state
        route_collections[route].set_visible(state)
        draw_trains(time.time())

    def on_check(label):
        set_route(label, not visible[label])

    checks.on_clicked(on_check)

    def set_all(state):
        for i, route in enumerate(routes):
            if visible[route] != state:
                checks.set_active(i)   # fires on_check, which updates the map

    all_ax = fig.add_axes([0.77, 0.05, 0.1, 0.045])
    none_ax = fig.add_axes([0.89, 0.05, 0.1, 0.045])
    btn_all = Button(all_ax, "All", color="#222222", hovercolor="#444444")
    btn_none = Button(none_ax, "None", color="#222222", hovercolor="#444444")
    btn_all.label.set_color("white")
    btn_none.label.set_color("white")
    btn_all.on_clicked(lambda _e: set_all(True))
    btn_none.on_clicked(lambda _e: set_all(False))

    # Latest feed data; trains are placed from this every frame.
    state = {"models": [], "fetched_at": 0.0, "error": None}

    def refresh_feeds():
        try:
            state["models"], state["fetched_at"] = fetch_model(stops, trip_stops)
            state["error"] = None
        except Exception as exc:
            state["error"] = str(exc)

    def draw_trains(now):
        models = [m for m in state["models"] if visible.get(m["route"], False)]
        placed = spread_overlaps([(m["route"], *position(m, now)) for m in models])
        if placed:
            pts = np.array([(lon, lat) for _, lat, lon in placed])
            colors = [ROUTE_COLORS.get(r, "#ffffff") for r, _, _ in placed]
            trains_plot.set_offsets(pts)
            trains_plot.set_facecolor(colors)
        else:
            trains_plot.set_offsets(np.empty((0, 2)))
        shown = sum(1 for r in visible if visible[r])
        if state["error"]:
            title.set_text(f"Live data unavailable: {state['error']}")
        else:
            title.set_text(f"Trains in service: {len(placed)}  ·  {shown} lines shown  ·  "
                           f"updated {time.strftime('%I:%M:%S %p', time.localtime(state['fetched_at']))}")
        fig.canvas.draw_idle()

    if args.once:
        refresh_feeds()
        draw_trains(time.time())
        out = HERE / "live_map.png"
        fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight")
        print(f"Saved {out}")
        return

    def frame(_i):
        now = time.time()
        if now - state["fetched_at"] >= FEED_REFRESH_SECONDS:
            refresh_feeds()     # every 30 seconds, the trains re-anchor to the latest predictions
        draw_trains(now)
        return trains_plot, title

    # Keep a reference so the animation isn't garbage-collected.
    anim = FuncAnimation(fig, frame, interval=FRAME_MS, cache_frame_data=False)
    refresh_feeds()
    plt.show()


if __name__ == "__main__":
    main()
