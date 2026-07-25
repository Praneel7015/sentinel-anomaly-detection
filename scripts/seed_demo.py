"""
Seed script -- fires realistic access events at the SENTINEL API.
Run: python scripts/seed_demo.py
Uses only stdlib, no extra installs needed.
"""
import json
import random
import uuid
import hashlib
from datetime import datetime, timezone, timedelta
import urllib.request
import urllib.error
import sys

API = "https://sentinel-anomaly-detection.onrender.com/api/score"

rng = random.Random(42)

USERS = ["user_042", "user_019", "user_107", "user_203", "user_055"]
SERVICES = ["svc_payroll", "svc_reporting"]
DEVICES = ["device_floor3_01", "device_dc_02"]
ALL_ENTITIES = USERS + SERVICES + DEVICES

COHORTS = {
    "user_042": "finance_analyst", "user_019": "engineer",
    "user_107": "hr_manager", "user_203": "engineer",
    "user_055": "finance_analyst", "svc_payroll": "service_account",
    "svc_reporting": "service_account", "device_floor3_01": "plc_gateway",
    "device_dc_02": "plc_gateway",
}

ENTITY_TYPES = {
    "user_042": "user", "user_019": "user", "user_107": "user",
    "user_203": "user", "user_055": "user",
    "svc_payroll": "service_account", "svc_reporting": "service_account",
    "device_floor3_01": "edge_device", "device_dc_02": "edge_device",
}

GEO = [
    ("DE", "Berlin", 52.52, 13.40),
    ("US", "New York", 40.71, -74.01),
    ("GB", "London", 51.51, -0.13),
    ("IN", "Mumbai", 19.08, 72.88),
    ("SG", "Singapore", 1.35, 103.82),
]

RESOURCES = [
    ("/app/dashboard",     "endpoint"),
    ("/api/health",        "endpoint"),
    ("/reports/monthly",   "file"),
    ("/hr/salaries",       "file"),
    ("/legal/contracts",   "file"),
    ("/finance/payroll",   "file"),
    ("/admin/passwords",   "endpoint"),
    ("/api/export/users",  "endpoint"),
    ("/db/prod/backup",    "file"),
    ("port:445",           "port"),
    ("port:22",            "port"),
]

OS_VARIANTS = [
    ("Windows", "10.0.19041"),
    ("Linux",   "Ubuntu 22.04"),
    ("macOS",   "13.4"),
]

def fingerprint(os: str, ver: str, mac: str, proto: str) -> str:
    raw = f"{os}|{ver}|{mac}|{proto}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]

def rand_mac() -> str:
    return ":".join(f"{rng.randint(0,255):02x}" for _ in range(6))

def make_event(entity_id: str, geo_idx: int, resource: str, resource_type: str,
               auth_result: str, bytes_xfer: int, ts: datetime,
               extra_ip: str | None = None) -> dict:
    country, city, lat, lon = GEO[geo_idx]
    os_name, os_ver = rng.choice(OS_VARIANTS)
    mac = rand_mac()
    proto = rng.choice(["https", "ssh", "rdp", "smb"])
    fp = fingerprint(os_name, os_ver, mac, proto)
    ip = extra_ip or f"10.{rng.randint(0,5)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
    return {
        "event_id":          uuid.uuid4().hex[:16],
        "episode_id":        None,
        "entity_id":         entity_id,
        "entity_type":       ENTITY_TYPES[entity_id],
        "cohort":            COHORTS[entity_id],
        "timestamp":         ts.isoformat(),
        "source_ip":         ip,
        "geo_country":       country,
        "geo_city":          city,
        "geo_lat":           lat + rng.uniform(-0.5, 0.5),
        "geo_lon":           lon + rng.uniform(-0.5, 0.5),
        "resource_accessed": resource,
        "resource_type":     resource_type,
        "auth_method":       rng.choice(["password", "token", "mfa_push"]),
        "auth_result":       auth_result,
        "session_duration_s": rng.uniform(30, 3600),
        "command_sequence":  [],
        "device_os":         os_name,
        "device_os_version": os_ver,
        "device_mac":        mac,
        "device_protocol":   proto,
        "device_fingerprint": fp,
        "bytes_transferred": bytes_xfer,
        "split":             "test",
    }

def post_event(event: dict) -> dict | None:
    data = json.dumps(event).encode()
    req = urllib.request.Request(
        API, data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        print(f"  HTTP {e.code}: {body}")
    except Exception as ex:
        print(f"  Error: {ex}")
    return None

def main():
    now = datetime.now(timezone.utc)
    total = 0
    alerts = 0

    print(f"Seeding SENTINEL at {API}")
    print("=" * 60)

    # 1. Normal baseline -- 40 routine events
    print("\n[1/3] Normal baseline (40 events)...")
    for i in range(40):
        entity = rng.choice(ALL_ENTITIES)
        res, res_type = rng.choice(RESOURCES[:6])
        geo = rng.randint(0, 4)
        ts = now - timedelta(minutes=rng.randint(10, 480))
        result = post_event(make_event(entity, geo, res, res_type, "success",
                                       rng.randint(512, 8192), ts))
        if result:
            total += 1
            risk = result.get("risk_score", 0)
            is_alert = result.get("is_alert", False)
            if is_alert:
                alerts += 1
            marker = "ALERT" if is_alert else "     "
            print(f"  {i+1:02d}. {entity:22s} risk={risk:5.1f} {marker}")

    # 2. Brute force attack on user_042 (8 failures, then success)
    print("\n[2/3] Brute force attack (9 events on user_042)...")
    for i in range(8):
        ts = now - timedelta(minutes=20 - i * 2)
        result = post_event(make_event(
            "user_042", 0, "/admin/passwords", "endpoint",
            "failure", 0, ts, extra_ip="192.168.99.5"
        ))
        if result:
            total += 1
            risk = result.get("risk_score", 0)
            is_alert = result.get("is_alert", False)
            if is_alert:
                alerts += 1
            print(f"  {i+1:02d}. user_042 brute_force failure  risk={risk:5.1f} {'ALERT' if is_alert else ''}")
    # final success after brute
    result = post_event(make_event(
        "user_042", 0, "/admin/passwords", "endpoint",
        "success", 0, now - timedelta(minutes=2), extra_ip="192.168.99.5"
    ))
    if result:
        total += 1
        risk = result.get("risk_score", 0)
        is_alert = result.get("is_alert", False)
        if is_alert:
            alerts += 1
        print(f"  09. user_042 brute_force SUCCESS  risk={risk:5.1f} {'ALERT' if is_alert else ''}")

    # 3. Lateral movement -- svc_payroll hops across sensitive resources
    print("\n[3/3] Lateral movement (5 events on svc_payroll)...")
    lateral_resources = [
        ("/finance/payroll",  "file",     1_048_576),
        ("/db/prod/backup",   "file",     8_388_608),
        ("/api/export/users", "endpoint", 2_097_152),
        ("port:445",          "port",           0),
        ("port:22",           "port",           0),
    ]
    for i, (res, rt, byt) in enumerate(lateral_resources):
        ts = now - timedelta(minutes=10 - i * 2)
        result = post_event(make_event(
            "svc_payroll", 1, res, rt, "success", byt, ts,
            extra_ip="10.0.1.88"
        ))
        if result:
            total += 1
            risk = result.get("risk_score", 0)
            is_alert = result.get("is_alert", False)
            if is_alert:
                alerts += 1
            print(f"  {i+1:02d}. svc_payroll  {res:30s} risk={risk:5.1f} {'ALERT' if is_alert else ''}")

    print("\n" + "=" * 60)
    print(f"Done. {total} events sent -> {alerts} alerts raised.")
    print("Open https://sentinel-anomaly-detection-phi.vercel.app")

if __name__ == "__main__":
    main()
