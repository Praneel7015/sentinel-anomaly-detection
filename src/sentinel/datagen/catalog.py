"""World model: cohort definitions, geo table, IP allocator, and device fingerprint helpers.

Everything that multiple generator modules need to agree on lives here.
No randomness is introduced at this level - callers pass their own RNG.
"""
from __future__ import annotations

import hashlib
import ipaddress
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "COHORTS",
    "GEO_SITES",
    "CohortDef",
    "GeoSite",
    "device_fingerprint",
    "ip_for_site",
    "random_mac",
]


# ---------------------------------------------------------------------------
# Geo sites - stable subnets so the same entity always looks like it comes
# from the same /24.  Coordinates are real cities; subnet is fictional.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeoSite:
    city: str
    country: str          # ISO 3166-1 alpha-2
    lat: float
    lon: float
    subnet: str           # e.g. "10.11.0.0/16" - IPs allocated from here


GEO_SITES: list[GeoSite] = [
    GeoSite("New York",       "US",  40.71, -74.01, "10.10.0.0/16"),
    GeoSite("Chicago",        "US",  41.88, -87.63, "10.11.0.0/16"),
    GeoSite("San Francisco",  "US",  37.77,-122.42, "10.12.0.0/16"),
    GeoSite("Houston",        "US",  29.76, -95.37, "10.13.0.0/16"),
    GeoSite("London",         "GB",  51.51,  -0.13, "10.20.0.0/16"),
    GeoSite("Frankfurt",      "DE",  50.11,   8.68, "10.21.0.0/16"),
    GeoSite("Paris",          "FR",  48.85,   2.35, "10.22.0.0/16"),
    GeoSite("Amsterdam",      "NL",  52.37,   4.90, "10.23.0.0/16"),
    GeoSite("Singapore",      "SG",   1.35, 103.82, "10.30.0.0/16"),
    GeoSite("Tokyo",          "JP",  35.68, 139.69, "10.31.0.0/16"),
    GeoSite("Sydney",         "AU", -33.87, 151.21, "10.32.0.0/16"),
    GeoSite("Mumbai",         "IN",  19.08,  72.88, "10.33.0.0/16"),
    GeoSite("Toronto",        "CA",  43.65, -79.38, "10.40.0.0/16"),
    GeoSite("Sao Paulo",      "BR", -23.55, -46.63, "10.41.0.0/16"),
    GeoSite("Dubai",          "AE",  25.20,  55.27, "10.42.0.0/16"),
    GeoSite("Cape Town",      "ZA", -33.93,  18.42, "10.43.0.0/16"),
    GeoSite("Detroit",        "US",  42.33, -83.05, "10.14.0.0/16"),  # plant / OT
    GeoSite("Houston OT",     "US",  29.80, -95.40, "10.15.0.0/16"),  # refinery
    GeoSite("Dallas",         "US",  32.78, -96.80, "10.16.0.0/16"),  # retail / POS
]

# Index by city name for quick lookup
_SITE_BY_CITY: dict[str, GeoSite] = {s.city: s for s in GEO_SITES}


def ip_for_site(site: GeoSite, rng: np.random.Generator) -> str:
    """Return a random IP from the site's /16 subnet (host part 1-254)."""
    network = ipaddress.IPv4Network(site.subnet, strict=False)
    # network address + random offset within the /16 (avoid .0 and .255)
    offset = int(rng.integers(1, 65534))
    return str(network.network_address + offset)


def random_mac(rng: np.random.Generator) -> str:
    """Return a random locally-administered unicast MAC."""
    raw = rng.integers(0, 256, size=6, dtype=np.uint8)
    raw[0] = (raw[0] & 0xFE) | 0x02   # locally administered, unicast
    return ":".join(f"{b:02x}" for b in raw)


def device_fingerprint(os: str, version: str, mac: str, protocol: str) -> str:
    """Stable 16-char hex fingerprint of the four device identity fields."""
    blob = f"{os}|{version}|{mac}|{protocol}".encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Resource pools - cohort-specific, with schema-compliant prefixes
# ---------------------------------------------------------------------------

_FINANCE_RESOURCES = [
    ("fs:/finance/q1.xlsx", "file"), ("fs:/finance/q2.xlsx", "file"),
    ("fs:/finance/q3.xlsx", "file"), ("fs:/finance/q4.xlsx", "file"),
    ("fs:/finance/budget_2026.xlsx", "file"), ("fs:/finance/payroll.csv", "file"),
    ("api:/v1/payments", "endpoint"), ("api:/v1/invoices", "endpoint"),
    ("api:/v1/accounts", "endpoint"), ("api:/v1/reports/finance", "endpoint"),
    ("fs:/shared/contracts/", "file"), ("fs:/audit/logs/", "file"),
]

_HR_RESOURCES = [
    ("fs:/hr/employee_records.db", "file"), ("fs:/hr/compensation.xlsx", "file"),
    ("fs:/hr/performance_reviews/", "file"), ("fs:/hr/onboarding/", "file"),
    ("api:/v1/hr/employees", "endpoint"), ("api:/v1/hr/payroll", "endpoint"),
    ("api:/v1/hr/benefits", "endpoint"), ("fs:/shared/policies/", "file"),
    ("fs:/hr/org_chart.pptx", "file"),
]

_DEVOPS_RESOURCES = [
    ("api:/v1/deploy", "endpoint"), ("api:/v1/ci/pipelines", "endpoint"),
    ("api:/v1/k8s/pods", "endpoint"), ("api:/v1/k8s/deployments", "endpoint"),
    ("port:22", "port"), ("port:443", "port"), ("port:6443", "port"),
    ("fs:/configs/prod/", "file"), ("fs:/configs/staging/", "file"),
    ("api:/v1/secrets", "endpoint"), ("api:/v1/logs", "endpoint"),
    ("api:/v1/monitoring", "endpoint"), ("port:9090", "port"),
]

_DB_ADMIN_RESOURCES = [
    ("port:5432", "port"), ("port:3306", "port"), ("port:1521", "port"),
    ("port:27017", "port"), ("api:/v1/db/query", "endpoint"),
    ("api:/v1/db/backup", "endpoint"), ("api:/v1/db/restore", "endpoint"),
    ("fs:/db/dumps/", "file"), ("fs:/db/configs/", "file"),
    ("api:/v1/db/users", "endpoint"), ("api:/v1/db/audit", "endpoint"),
]

_SALES_RESOURCES = [
    ("api:/v1/crm/leads", "endpoint"), ("api:/v1/crm/opportunities", "endpoint"),
    ("api:/v1/crm/accounts", "endpoint"), ("fs:/sales/proposals/", "file"),
    ("fs:/sales/contracts/", "file"), ("api:/v1/pricing", "endpoint"),
    ("api:/v1/reports/sales", "endpoint"), ("fs:/marketing/assets/", "file"),
    ("api:/v1/crm/dashboard", "endpoint"),
]

_SUPPORT_RESOURCES = [
    ("api:/v1/tickets", "endpoint"), ("api:/v1/tickets/search", "endpoint"),
    ("api:/v1/users/lookup", "endpoint"), ("fs:/kb/articles/", "file"),
    ("api:/v1/diagnostics", "endpoint"), ("api:/v1/logs/search", "endpoint"),
    ("port:3389", "port"), ("api:/v1/remote_assist", "endpoint"),
]

_BACKUP_SVC_RESOURCES = [
    ("fs:/backups/daily/", "file"), ("fs:/backups/weekly/", "file"),
    ("fs:/finance/", "file"), ("fs:/hr/", "file"), ("fs:/db/dumps/", "file"),
    ("api:/v1/db/backup", "endpoint"), ("fs:/configs/", "file"),
    ("api:/v1/storage/upload", "endpoint"),
]

_CI_RUNNER_RESOURCES = [
    ("api:/v1/ci/pipelines", "endpoint"), ("api:/v1/deploy", "endpoint"),
    ("port:22", "port"), ("api:/v1/k8s/deployments", "endpoint"),
    ("api:/v1/registry/push", "endpoint"), ("api:/v1/secrets", "endpoint"),
    ("fs:/artifacts/", "file"), ("api:/v1/test-results", "endpoint"),
]

_ETL_RESOURCES = [
    ("fs:/data/raw/", "file"), ("fs:/data/processed/", "file"),
    ("api:/v1/etl/jobs", "endpoint"), ("port:5432", "port"),
    ("port:9000", "port"), ("api:/v1/warehouse/load", "endpoint"),
    ("fs:/finance/", "file"), ("fs:/hr/", "file"),
]

_PLANT_GW_RESOURCES = [
    ("func:valve_setpoint", "device_function"), ("func:pump_speed", "device_function"),
    ("func:temp_sensor_read", "device_function"), ("func:pressure_read", "device_function"),
    ("func:flow_meter", "device_function"), ("func:emergency_stop", "device_function"),
    ("port:502", "port"),  # modbus
]

_POS_RESOURCES = [
    ("func:card_swipe", "device_function"), ("func:receipt_print", "device_function"),
    ("func:drawer_open", "device_function"), ("api:/v1/pos/transaction", "endpoint"),
    ("api:/v1/pos/inventory", "endpoint"), ("port:8080", "port"),
]

_HVAC_RESOURCES = [
    ("func:temp_setpoint", "device_function"), ("func:fan_speed", "device_function"),
    ("func:filter_status", "device_function"), ("func:zone_control", "device_function"),
    ("func:energy_meter", "device_function"),
]

_RTU_RESOURCES = [
    ("func:breaker_status", "device_function"), ("func:voltage_read", "device_function"),
    ("func:current_read", "device_function"), ("func:relay_control", "device_function"),
    ("func:fault_register", "device_function"), ("port:502", "port"),
]


# ---------------------------------------------------------------------------
# Command sequences - per-cohort base Markov chain encoded as transition probs
# The attacker chain is used by lateral_movement injector.
# ---------------------------------------------------------------------------

BASE_COMMANDS: dict[str, list[str]] = {
    "finance_analyst":   ["open_file", "read", "export", "print", "close"],
    "hr_generalist":     ["search_employee", "view_record", "edit_record", "save", "export"],
    "devops_engineer":   ["ssh_connect", "git_pull", "build", "deploy", "check_logs"],
    "db_admin":          ["connect_db", "query", "explain", "vacuum", "backup"],
    "sales_field":       ["open_crm", "search_lead", "update_opportunity", "send_email", "sync"],
    "support_desk":      ["open_ticket", "search_kb", "remote_connect", "run_diagnostic", "close_ticket"],
    "backup_service":    ["scan_dirs", "compress", "transfer", "verify", "cleanup"],
    "ci_runner":         ["checkout", "build", "test", "package", "deploy"],
    "etl_job":           ["extract", "transform", "validate", "load", "report"],
    "plant_gateway":     ["read_sensor", "write_setpoint", "heartbeat", "alarm_check"],
    "pos_terminal":      ["init_session", "process_transaction", "print_receipt", "close_session"],
    "hvac_controller":   ["read_temp", "adjust_setpoint", "fan_control", "log_status"],
    "substation_rtu":    ["read_voltage", "read_current", "check_relay", "log_event"],
}

# Attacker chain used by lateral_movement: recon -> enumerate -> priv_esc -> remote_exec
ATTACKER_COMMANDS: list[str] = [
    "nmap_scan", "enum_shares", "enum_users", "enum_services",
    "exploit_cve", "add_local_admin", "dump_creds",
    "psexec_connect", "remote_cmd", "exfil_data",
]

ATTACKER_CHAIN_STAGES: dict[str, str] = {
    "nmap_scan": "recon", "enum_shares": "recon", "enum_users": "recon",
    "enum_services": "recon", "exploit_cve": "escalation",
    "add_local_admin": "escalation", "dump_creds": "escalation",
    "psexec_connect": "action_on_objective", "remote_cmd": "action_on_objective",
    "exfil_data": "action_on_objective",
}


# ---------------------------------------------------------------------------
# Cohort definitions
# ---------------------------------------------------------------------------

@dataclass
class CohortDef:
    name: str
    entity_type: str                    # user | service_account | edge_device
    resources: list[tuple[str, str]]    # (resource_accessed, resource_type)
    protocols: list[str]
    auth_methods: list[str]
    privileged: bool                    # does this cohort run command sequences?
    # Circadian shape: list of (mean_hour, weight) tuples for the GMM
    circadian_peaks: list[tuple[float, float]]
    circadian_sigma: float              # std-dev of each peak (hours)
    weekday_weights: list[float]        # Mon-Sun, len=7
    home_sites: list[str]               # city names from GEO_SITES
    secondary_sites: list[str]          # occasional travel destinations
    bytes_mu: float                     # lognormal mean (log-scale)
    bytes_sigma: float                  # lognormal sigma (log-scale)
    session_duration_mu: float          # lognormal mean (log-scale, seconds)
    session_duration_sigma: float
    base_commands: list[str]
    weight: float = 1.0                 # relative sampling weight within entity_type


COHORTS: dict[str, CohortDef] = {
    "finance_analyst": CohortDef(
        name="finance_analyst", entity_type="user",
        resources=_FINANCE_RESOURCES,
        protocols=["https"],
        auth_methods=["mfa_push", "password"],
        privileged=False,
        circadian_peaks=[(9.0, 0.4), (14.0, 0.6)],
        circadian_sigma=1.5,
        weekday_weights=[1.0, 1.0, 1.0, 1.0, 0.8, 0.1, 0.05],
        home_sites=["New York", "Chicago", "London", "Frankfurt"],
        secondary_sites=["London", "New York", "Frankfurt", "Paris"],
        bytes_mu=9.5, bytes_sigma=1.8,
        session_duration_mu=6.5, session_duration_sigma=0.8,
        base_commands=BASE_COMMANDS["finance_analyst"],
    ),
    "hr_generalist": CohortDef(
        name="hr_generalist", entity_type="user",
        resources=_HR_RESOURCES,
        protocols=["https"],
        auth_methods=["mfa_push", "password"],
        privileged=False,
        circadian_peaks=[(9.5, 0.5), (13.5, 0.5)],
        circadian_sigma=1.2,
        weekday_weights=[1.0, 1.0, 1.0, 1.0, 0.9, 0.05, 0.02],
        home_sites=["New York", "Chicago", "London"],
        secondary_sites=["Chicago", "New York"],
        bytes_mu=8.5, bytes_sigma=1.5,
        session_duration_mu=6.8, session_duration_sigma=0.7,
        base_commands=BASE_COMMANDS["hr_generalist"],
    ),
    "devops_engineer": CohortDef(
        name="devops_engineer", entity_type="user",
        resources=_DEVOPS_RESOURCES,
        protocols=["ssh", "https"],
        auth_methods=["certificate", "token", "mfa_push"],
        privileged=True,
        circadian_peaks=[(10.0, 0.3), (15.0, 0.4), (21.0, 0.3)],
        circadian_sigma=2.0,
        weekday_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 0.3],
        home_sites=["San Francisco", "New York", "London", "Amsterdam"],
        secondary_sites=["San Francisco", "Amsterdam", "Singapore"],
        bytes_mu=10.0, bytes_sigma=2.0,
        session_duration_mu=7.5, session_duration_sigma=1.0,
        base_commands=BASE_COMMANDS["devops_engineer"],
    ),
    "db_admin": CohortDef(
        name="db_admin", entity_type="user",
        resources=_DB_ADMIN_RESOURCES,
        protocols=["ssh", "https"],
        auth_methods=["certificate", "password", "mfa_push"],
        privileged=True,
        circadian_peaks=[(8.0, 0.35), (14.0, 0.35), (22.0, 0.30)],
        circadian_sigma=1.8,
        weekday_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.4],
        home_sites=["New York", "San Francisco", "Frankfurt"],
        secondary_sites=["London", "Frankfurt"],
        bytes_mu=11.0, bytes_sigma=2.2,
        session_duration_mu=7.8, session_duration_sigma=1.1,
        base_commands=BASE_COMMANDS["db_admin"],
    ),
    "sales_field": CohortDef(
        name="sales_field", entity_type="user",
        resources=_SALES_RESOURCES,
        protocols=["https"],
        auth_methods=["mfa_push", "password", "token"],
        privileged=False,
        circadian_peaks=[(8.0, 0.25), (12.0, 0.25), (17.0, 0.50)],
        circadian_sigma=2.5,
        weekday_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.3],
        home_sites=["New York", "Chicago", "Houston", "Dallas", "Toronto"],
        secondary_sites=["London", "Singapore", "Dubai", "Sao Paulo", "Tokyo",
                         "San Francisco", "New York", "Chicago"],
        bytes_mu=8.0, bytes_sigma=1.5,
        session_duration_mu=5.8, session_duration_sigma=0.9,
        base_commands=BASE_COMMANDS["sales_field"],
    ),
    "support_desk": CohortDef(
        name="support_desk", entity_type="user",
        resources=_SUPPORT_RESOURCES,
        protocols=["https", "rdp"],
        auth_methods=["password", "mfa_push"],
        privileged=True,
        circadian_peaks=[(9.0, 0.5), (14.0, 0.5)],
        circadian_sigma=1.3,
        weekday_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.1],
        home_sites=["Chicago", "New York", "London", "Mumbai"],
        secondary_sites=["New York", "Chicago"],
        bytes_mu=8.0, bytes_sigma=1.4,
        session_duration_mu=6.2, session_duration_sigma=0.9,
        base_commands=BASE_COMMANDS["support_desk"],
    ),
    "backup_service": CohortDef(
        name="backup_service", entity_type="service_account",
        resources=_BACKUP_SVC_RESOURCES,
        protocols=["https", "ssh"],
        auth_methods=["certificate", "token"],
        privileged=True,
        circadian_peaks=[(2.0, 0.7), (14.0, 0.3)],
        circadian_sigma=0.5,
        weekday_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        home_sites=["New York", "London", "Frankfurt"],
        secondary_sites=[],
        bytes_mu=13.0, bytes_sigma=1.5,
        session_duration_mu=8.5, session_duration_sigma=0.6,
        base_commands=BASE_COMMANDS["backup_service"],
    ),
    "ci_runner": CohortDef(
        name="ci_runner", entity_type="service_account",
        resources=_CI_RUNNER_RESOURCES,
        protocols=["https", "ssh"],
        auth_methods=["token", "certificate"],
        privileged=True,
        circadian_peaks=[(10.0, 0.4), (14.0, 0.3), (16.0, 0.3)],
        circadian_sigma=1.5,
        weekday_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.2],
        home_sites=["San Francisco", "New York", "London"],
        secondary_sites=[],
        bytes_mu=11.5, bytes_sigma=1.8,
        session_duration_mu=7.2, session_duration_sigma=0.8,
        base_commands=BASE_COMMANDS["ci_runner"],
    ),
    "etl_job": CohortDef(
        name="etl_job", entity_type="service_account",
        resources=_ETL_RESOURCES,
        protocols=["https"],
        auth_methods=["token", "certificate"],
        privileged=True,
        circadian_peaks=[(3.0, 0.5), (11.0, 0.3), (23.0, 0.2)],
        circadian_sigma=0.4,
        weekday_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        home_sites=["New York", "Frankfurt", "Singapore"],
        secondary_sites=[],
        bytes_mu=12.5, bytes_sigma=2.0,
        session_duration_mu=7.0, session_duration_sigma=0.6,
        base_commands=BASE_COMMANDS["etl_job"],
    ),
    "plant_gateway": CohortDef(
        name="plant_gateway", entity_type="edge_device",
        resources=_PLANT_GW_RESOURCES,
        protocols=["modbus", "mqtt"],
        auth_methods=["certificate"],
        privileged=False,
        circadian_peaks=[(12.0, 1.0)],  # near-uniform - wide sigma
        circadian_sigma=12.0,
        weekday_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        home_sites=["Detroit", "Houston OT"],
        secondary_sites=[],
        bytes_mu=5.5, bytes_sigma=0.8,
        session_duration_mu=3.5, session_duration_sigma=0.4,
        base_commands=BASE_COMMANDS["plant_gateway"],
    ),
    "pos_terminal": CohortDef(
        name="pos_terminal", entity_type="edge_device",
        resources=_POS_RESOURCES,
        protocols=["https"],
        auth_methods=["certificate", "token"],
        privileged=False,
        circadian_peaks=[(10.0, 0.15), (12.0, 0.30), (18.0, 0.55)],
        circadian_sigma=1.5,
        weekday_weights=[0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 0.8],
        home_sites=["Dallas", "Houston", "Chicago"],
        secondary_sites=[],
        bytes_mu=6.5, bytes_sigma=1.0,
        session_duration_mu=3.0, session_duration_sigma=0.5,
        base_commands=BASE_COMMANDS["pos_terminal"],
    ),
    "hvac_controller": CohortDef(
        name="hvac_controller", entity_type="edge_device",
        resources=_HVAC_RESOURCES,
        protocols=["mqtt"],
        auth_methods=["certificate"],
        privileged=False,
        circadian_peaks=[(12.0, 1.0)],
        circadian_sigma=12.0,
        weekday_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        home_sites=["New York", "Chicago", "London"],
        secondary_sites=[],
        bytes_mu=4.5, bytes_sigma=0.6,
        session_duration_mu=2.5, session_duration_sigma=0.3,
        base_commands=BASE_COMMANDS["hvac_controller"],
    ),
    "substation_rtu": CohortDef(
        name="substation_rtu", entity_type="edge_device",
        resources=_RTU_RESOURCES,
        protocols=["modbus"],
        auth_methods=["certificate"],
        privileged=False,
        circadian_peaks=[(12.0, 1.0)],
        circadian_sigma=12.0,
        weekday_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        home_sites=["Houston OT", "Detroit"],
        secondary_sites=[],
        bytes_mu=4.0, bytes_sigma=0.6,
        session_duration_mu=2.0, session_duration_sigma=0.3,
        base_commands=BASE_COMMANDS["substation_rtu"],
    ),
}

# Which cohorts map to which entity_type
COHORTS_BY_TYPE: dict[str, list[str]] = {
    "user": ["finance_analyst", "hr_generalist", "devops_engineer", "db_admin",
             "sales_field", "support_desk"],
    "service_account": ["backup_service", "ci_runner", "etl_job"],
    "edge_device": ["plant_gateway", "pos_terminal", "hvac_controller", "substation_rtu"],
}

# OS/version pool - realistic spread
OS_POOL: list[tuple[str, str]] = [
    ("Windows", "10"), ("Windows", "11"), ("Windows", "10"),
    ("macOS", "14.4"), ("macOS", "13.6"), ("macOS", "14.2"),
    ("Ubuntu", "22.04"), ("Ubuntu", "24.04"), ("RHEL", "9.3"),
    ("Debian", "12"), ("Windows Server", "2022"), ("Windows Server", "2019"),
    ("Android", "14"), ("iOS", "17"),
    ("Firmware", "3.1.2"), ("Firmware", "4.0.0"), ("Firmware", "2.9.7"),
]
