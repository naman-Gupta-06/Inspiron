# ============================================================
# api/server.py
# FastAPI & Native WebSockets bridge for React Heimdall Dashboard.
# ============================================================

import logging
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from database.alert_db import fetch_all_stations
from state import fleet_state

# Disable strict uvicorn access logs to keep the terminal clean for your alerts
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(title="Heimdall API")

# Allow CORS so your React frontend can connect seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WEBSOCKET MANAGER ───────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, event_name: str, data: dict):
        """Sends a structured JSON payload to all connected React clients."""
        payload = {"event": event_name, "data": data}
        for connection in self.active_connections:
            try:
                await connection.send_json(payload)
            except Exception:
                pass # Client disconnected abruptly

manager = ConnectionManager()

# ── 1. REST API: Initial Load ───────────────────────────────────────────────

@app.get("/api/stations")
def get_stations():
    """Frontend calls this ONCE when the map loads to get fixed infrastructure."""
    return {"status": "success", "data": fetch_all_stations()}

@app.get("/api/fleet_state")
def get_initial_fleet_state():
    """Fetch the exact current position of all drones on load."""
    return {"status": "success", "data": fleet_state.get_all_telemetry()}

# ── 2. Internal Webhooks (Backend to Server) ────────────────────────────────
# These receive synchronous requests from detector.py and priority.py 
# and broadcast them asynchronously to the frontend.

@app.post("/internal/alert")
async def handle_internal_alert(alert_data: Dict[str, Any]):
    await manager.broadcast('new_alert', alert_data)
    
    msg = f"🚨 NEW INCIDENT: {alert_data.get('incident_type', '').upper()} detected at {alert_data.get('latitude', 0):.4f}, {alert_data.get('longitude', 0):.4f}"
    await manager.broadcast('system_log', {"message": msg, "level": "WARNING"})
    return {"status": "broadcasted"}

@app.post("/internal/dispatch")
async def handle_internal_dispatch(dispatch_data: Dict[str, Any]):
    await manager.broadcast('drone_dispatched', dispatch_data)
    
    msg = f"🚁 DISPATCH: Drone {dispatch_data.get('drone_id')} en route to Alert {dispatch_data.get('alert_id')} (ETA: {dispatch_data.get('eta_seconds')}s)"
    await manager.broadcast('system_log', {"message": msg, "level": "SUCCESS"})
    return {"status": "broadcasted"}

@app.post("/internal/log")
async def handle_internal_log(log_data: Dict[str, Any]):
    await manager.broadcast('system_log', log_data)
    return {"status": "broadcasted"}

# ── 3. WEBSOCKETS: The Live Firehose ────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Single WebSocket connection for the React frontend to listen to updates
    and push live drone telemetry.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Receive data from React
            incoming = await websocket.receive_json()
            
            # If React is sending drone coordinates
            if incoming.get("event") == "frontend_drone_update":
                data = incoming.get("data", {})
                drone_id = data.get('drone_id')
                lat = data.get('lat')
                lon = data.get('lon')
                
                # Force the backend to calculate the battery drain
                updated_state = fleet_state.update_telemetry(drone_id, lat, lon)
                
                # Attach backend truth to the broadcast
                data['battery'] = updated_state['battery']
                
                # Broadcast the updated coordinates to all clients
                await manager.broadcast('live_telemetry', data)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def run_server():
    print("🌐 Starting FastAPI & Native WebSockets on port 5000...")
    # Run Uvicorn programmatically on port 5000 to keep webhooks intact
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="error")