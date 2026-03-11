class ClientRegistry:

    def __init__(self):
        self.clients = {}

    def update(self, client_id, trust_score):

        if client_id not in self.clients:
            self.clients[client_id] = []

        self.clients[client_id].append(trust_score)

    def get_client_history(self, client_id):
        return self.clients.get(client_id, [])