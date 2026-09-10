from __future__ import annotations

from .utils import range_list

PARTITIONS={
 "AP_CHART_FIT":(10000,10063),
 "AP_CHART_SELECT":(10064,10127),
 "AP_WRITER_FIT":(10128,10191),
 "AP_WRITER_SELECT":(10192,10255),
 "AP_QUOTIENT":(10256,10319),
 "AP_DYNAMICS":(10320,10383),
 "AP_META_CONFIRM":(10384,10447),
 "AP_DISCOVERY_FINAL":(10448,10511),
 "REUSED_HISTORICAL_VALIDATION":(20000,20127),
 "FRESH_AUDIT":(90000,90499),
}

def seeds(name:str)->list[int]: return range_list(PARTITIONS[name])

def assert_roles()->None:
    dev=[PARTITIONS[k] for k in PARTITIONS if k.startswith("AP_")]
    used=set()
    for a,b in dev:
        s=set(range(a,b+1))
        if used&s: raise RuntimeError("V837AP_DATA_ROLE_OVERLAP")
        used|=s
    if used & set(range(*[PARTITIONS['FRESH_AUDIT'][0],PARTITIONS['FRESH_AUDIT'][1]+1])):
        raise RuntimeError("V837AP_FRESH_AUDIT_OVERLAP")
