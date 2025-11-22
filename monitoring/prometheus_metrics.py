# monitoring/prometheus_metrics.py
from prometheus_client import start_http_server, Gauge
import time

client_acc = Gauge('client_local_accuracy', 'Per-client validation accuracy', ['client_id'])
global_round = Gauge('fl_global_round', 'Global FL round number')

def start_metrics_server(port=9100):
    start_http_server(port)
    print(f'[metrics] Prometheus metrics exposed on :{port}')

if __name__ == '__main__':
    start_metrics_server()
    r = 0
    while True:
        r += 1
        global_round.set(r)
        client_acc.labels(client_id='client_1').set(0.7)
        time.sleep(10)
