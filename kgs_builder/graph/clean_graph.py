# ./kgs__builder/graph/clean_graph.py

from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def clear_graph(self):
        with self.driver.session() as session:
            session.write_transaction(self._clear_graph)

    @staticmethod
    def _clear_graph(tx):
        tx.run("MATCH (n) DETACH DELETE n")
        print("Graph cleared successfully.")