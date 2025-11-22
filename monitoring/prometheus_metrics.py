# monitoring/prometheus_metrics.py
from prometheus_client import start_http_server, Gauge
import time, random

client_acc = Gauge('client_local_accuracy', 'Per-client validation accuracy', ['client_id'])
global_round = Gauge('fl_global_round', 'Current FL round number')

if __name__ == '__main__':
    start_http_server(9101)
    r = 0
    while True:
        r += 1
        global_round.set(r)
        client_acc.labels(client_id='client_1').set(random.random())
        client_acc.labels(client_id='client_2').set(random.random())
        client_acc.labels(client_id='client_3').set(random.random())
        time.sleep(10)
