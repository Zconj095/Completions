import socket
import threading
import time
import json
from typing import Optional, Dict, Any

class Server:
    def __init__(self, host='0.0.0.0', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.clients = {}
        
    def start(self):
        """Start the server and listen for connections."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.running = True
            print(f"Server started on {self.host}:{self.port}")
            
            while self.running:
                try:
                    conn, addr = self.socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(conn, addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except socket.error:
                    break
                    
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            self.stop()
    
    def handle_client(self, conn, addr):
        """Handle individual client connections."""
        client_id = f"{addr[0]}:{addr[1]}"
        self.clients[client_id] = conn
        print(f"Client connected: {client_id}")
        
        try:
            while self.running:
                data = conn.recv(1024)
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode('utf-8'))
                    response = self.process_message(message, client_id)
                    conn.sendall(json.dumps(response).encode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Fallback for non-JSON data
                    conn.sendall(data)
                    
        except Exception as e:
            print(f"Error handling client {client_id}: {e}")
        finally:
            conn.close()
            if client_id in self.clients:
                del self.clients[client_id]
            print(f"Client disconnected: {client_id}")
    
    def process_message(self, message: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Process incoming messages and return response."""
        msg_type = message.get('type', 'unknown')
        
        if msg_type == 'ping':
            return {'type': 'pong', 'timestamp': time.time()}
        elif msg_type == 'echo':
            return {'type': 'echo_response', 'data': message.get('data')}
        else:
            return {'type': 'error', 'message': f'Unknown message type: {msg_type}'}
    
    def broadcast(self, message: Dict[str, Any]):
        """Send message to all connected clients."""
        data = json.dumps(message).encode('utf-8')
        for client_id, conn in list(self.clients.items()):
            try:
                conn.sendall(data)
            except:
                del self.clients[client_id]
    
    def stop(self):
        """Stop the server."""
        self.running = False
        if self.socket:
            self.socket.close()
        print("Server stopped")

class Client:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to the server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def send_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a message and receive response."""
        if not self.connected:
            print("Not connected to server")
            return None
            
        try:
            data = json.dumps(message).encode('utf-8')
            self.socket.sendall(data)
            
            response = self.socket.recv(1024)
            return json.loads(response.decode('utf-8'))
        except Exception as e:
            print(f"Error sending message: {e}")
            return None
    
    def ping(self) -> Optional[float]:
        """Send a ping and measure response time."""
        start_time = time.time()
        response = self.send_message({'type': 'ping'})
        
        if response and response.get('type') == 'pong':
            return time.time() - start_time
        return None
    
    def echo(self, data: str) -> Optional[str]:
        """Send echo message."""
        response = self.send_message({'type': 'echo', 'data': data})
        if response and response.get('type') == 'echo_response':
            return response.get('data')
        return None
    
    def disconnect(self):
        """Disconnect from server."""
        if self.socket:
            self.socket.close()
            self.connected = False
            print("Disconnected from server")

# Usage examples:
def run_server():
    server = Server()
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()

def run_client():
    client = Client()
    if client.connect():
        # Test ping
        ping_time = client.ping()
        if ping_time:
            print(f"Ping: {ping_time*1000:.2f}ms")
        
        # Test echo
        echo_response = client.echo("Hello, enhanced world!")
        print(f"Echo response: {echo_response}")
        
        client.disconnect()

# Uncomment to run:
# run_server()  # Run this first
# run_client()  # Run this second
