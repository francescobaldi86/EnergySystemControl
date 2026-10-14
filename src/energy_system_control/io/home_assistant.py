import requests
from energy_system_control.io.core import IOClient

class HomeAssistantClient(IOClient):
    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

    def get_state_safe(self, entity_id: str) -> dict:
        response = self.session.get(
            f"{self.url}/api/states/{entity_id}",
            timeout=5,
        )
        response.raise_for_status()
        r = response.json()
        if r is None or r['state'] in ['unknown', 'unavailable']:
            return None
        return r['state']

    def get_float_state(self, entity_id: str, default=0.0):
        s = self.get_state_safe(entity_id)
        if s is not None:
            try:
                return float(s)
            except ValueError:
                pass
        return default

    def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        other_info: dict = {},
    ) -> dict:
        json = {"entity_id": entity_id}
        json.update(other_info)
        response = self.session.post(
            f"{self.url}/api/services/{domain}/{service}",
            json=json,
            timeout=5,
        )
        response.raise_for_status()
        return response.json()