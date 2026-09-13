import httpx
from langchain_core.tools import tool

# In production this was a live internal API. The same client code is kept here;
# only the transport is swapped for an in-memory stub so the repo runs offline.
_STUB = {
    "APP-1001": {"status": "approved", "landlord": "Oak Street Properties"},
    "APP-1002": {"status": "screening", "landlord": "Harbor View LLC"},
    "APP-1003": {"status": "expired", "landlord": "Pinecrest Rentals"},
}


def _handler(request: httpx.Request) -> httpx.Response:
    app_id = request.url.path.rsplit("/", 1)[-1]
    if app_id in _STUB:
        return httpx.Response(200, json={"id": app_id, **_STUB[app_id]})
    return httpx.Response(404, json={"error": "not found"})


client = httpx.Client(base_url="http://status.internal", transport=httpx.MockTransport(_handler))


@tool
def get_application_status(application_id: str) -> str:
    """Look up the current status of a rental application by its exact id (e.g. APP-1042)."""
    r = client.get(f"/applications/{application_id}")
    if r.status_code == 404:
        return f"No application found with id {application_id}."
    d = r.json()
    return f"Application {d['id']} is {d['status']} (landlord: {d['landlord']})."
