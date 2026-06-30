import os

import pandas as pd
import requests


class MaterialsProjectCollector:
    def __init__(self, api_key: str = None):
        self.api_key = (
            api_key or os.environ.get("MP_API_KEY") or os.environ.get("MATERIALS_PROJECT_API_KEY")
        )

    def collect(
        self,
        elements: list = None,
        fields: list = None,
        max_results: int = 50000,
        num_chunks: int = None,
    ) -> pd.DataFrame:
        from mp_api.client import MPRester

        if not self.api_key:
            return pd.DataFrame()
        if num_chunks is None:
            num_chunks = min(max(1, max_results // 1000), 5)
        with MPRester(self.api_key) as mpr:
            docs = mpr.materials.summary.search(
                elements=elements or ["Li"],
                fields=fields
                or [
                    "material_id",
                    "formula_pretty",
                    "structure",
                    "formation_energy_per_atom",
                    "energy_above_hull",
                    "band_gap",
                    "volume",
                    "density",
                    "symmetry",
                    "is_stable",
                ],
                num_chunks=num_chunks,
            )
        return pd.DataFrame([d.dict() for d in docs])


class JARVISCollector:
    def collect(self, dataset_name: str = "dft_3d") -> pd.DataFrame:
        from jarvis.db.figshare import data

        return pd.DataFrame(data(dataset_name))


class OQMDCollector:
    def __init__(self, api_url: str = "https://oqmd.org/oqmdapi/formationenergy"):
        self.api_url = api_url

    def collect(self, limit: int = 100000, offset: int = 0) -> pd.DataFrame:
        all_entries = []
        batch_size = 1000
        local_offset = offset
        while len(all_entries) < limit:
            resp = requests.get(
                self.api_url,
                params={
                    "fields": "name,entry,delta_e,stability,unit_cell,band_gap",
                    "limit": min(batch_size, limit - len(all_entries)),
                    "offset": local_offset,
                },
                timeout=30,
            )
            if resp.status_code != 200:
                break
            data = resp.json().get("data", [])
            if not data:
                break
            all_entries.extend(data)
            local_offset += len(data)
        return pd.DataFrame(all_entries)


class AFLOWCollector:
    def collect(self, elements=None, max_results=50000) -> pd.DataFrame:
        entries = []
        page = 0
        while len(entries) < max_results:
            try:
                url = "https://aflow.org/API/aflux/"
                params = {
                    "species([*])": "Li",
                    "$paging": page * 100,
                    "$limit": min(100, max_results - len(entries)),
                    "$select": "auid,compound,Egap,enthalpy_formation_atom,nspecies",
                }
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code != 200:
                    break
                data = resp.json()
                if not data:
                    break
                entries.extend(data)
                page += 1
            except Exception:
                break
        return pd.DataFrame(entries)


SULFIDE_FILTERS = {
    "elements": ["Li", "S"],
    "exclude_elements": ["O"],
    "nelements_max": 5,
    "is_stable": True,
}

KNOWN_SULFIDES = [
    "Li6PS5Cl",
    "Li10GeP2S12",
    "Li3PS4",
    "Li7P3S11",
    "Li4SnS4",
    "Li6P2S5I",
]


class NOMADCollector:
    def collect(
        self, elements: list = None, page_size: int = 100, max_entries: int = 10000
    ) -> pd.DataFrame:
        url = "https://nomad-lab.eu/prod/v1/api/v1/entries/query"
        all_entries = []
        page_after_value = None
        while len(all_entries) < max_entries:
            body = {
                "query": {"elements": elements or ["Li", "S"]},
                "pagination": {"page_size": min(page_size, max_entries - len(all_entries))},
                "required": {"include": ["entry_id", "formula", "results.material.symmetry"]},
            }
            if page_after_value:
                body["pagination"]["page_after_value"] = page_after_value
            resp = requests.post(url, json=body, timeout=30)
            if resp.status_code != 200:
                break
            data = resp.json()
            entries = data.get("data", [])
            if not entries:
                break
            all_entries.extend(entries)
            pagination = data.get("pagination", {})
            page_after_value = pagination.get("next_page_after_value")
            if not page_after_value:
                break
        return pd.DataFrame(all_entries)
