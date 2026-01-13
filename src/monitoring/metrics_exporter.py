"""
Prometheus Metrics Exporter for RunPod Endpoints
Push metrics to Prometheus Pushgateway
"""
import os
import time
# import requests
# from typing import Dict, Any, Optional
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway
from functools import wraps

# Pushgateway URL
PUSHGATEWAY_URL = os.getenv('PUSHGATEWAY_URL', 'localhost:9091')

# Create registry
registry = CollectorRegistry()

# Define metrics

# 1️⃣ RELIABILITY Metrics
request_total = Counter(
    'endpoint_requests_total',
    'Total number of requests',
    ['endpoint', 'action', 'status'],
    registry=registry
)

cold_start_total = Counter(
    'endpoint_cold_starts_total',
    'Total number of cold starts',
    ['endpoint'],
    registry=registry
)

# 2️⃣ PERFORMANCE Metrics
request_duration = Histogram(
    'endpoint_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint', 'action'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry
)

inference_duration = Histogram(
    'endpoint_inference_duration_seconds',
    'Model inference duration in seconds',
    ['endpoint', 'action'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry
)

batch_size = Histogram(
    'endpoint_batch_size',
    'Batch size distribution',
    ['endpoint'],
    buckets=[1, 2, 5, 10, 20, 32, 50, 100],
    registry=registry
)

# 3️⃣ MODEL QUALITY Metrics
embedding_dimension = Gauge(
    'endpoint_embedding_dimension',
    'Embedding dimension',
    ['endpoint'],
    registry=registry
)

score_value = Histogram(
    'endpoint_score_value',
    'Cross-encoder score distribution',
    ['endpoint'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry
)

score_min = Gauge(
    'endpoint_score_min',
    'Minimum score in batch',
    ['endpoint'],
    registry=registry
)

score_max = Gauge(
    'endpoint_score_max',
    'Maximum score in batch',
    ['endpoint'],
    registry=registry
)

# 5️⃣ COST Metrics
cost_per_request = Gauge(
    'endpoint_cost_per_request',
    'Cost per request in USD',
    ['endpoint'],
    registry=registry
)

cost_per_1k = Gauge(
    'endpoint_cost_per_1k_requests',
    'Cost per 1000 requests in USD',
    ['endpoint'],
    registry=registry
)


class MetricsExporter:
    """Export metrics to Prometheus Pushgateway"""
    
    def __init__(self, endpoint_name: str, pushgateway_url: str = PUSHGATEWAY_URL):
        self.endpoint_name = endpoint_name
        self.pushgateway_url = pushgateway_url
        self.is_cold_start = True
    
    def push_metrics(self):
        """Push metrics to Pushgateway"""
        try:
            push_to_gateway(
                self.pushgateway_url,
                job=f'{self.endpoint_name}-metrics',
                registry=registry
            )
        except Exception as e:
            print(f"Failed to push metrics: {e}")
    
    def track_request(self, action: str, status: str, duration: float):
        """Track request metrics"""
        request_total.labels(
            endpoint=self.endpoint_name,
            action=action,
            status=status
        ).inc()
        
        request_duration.labels(
            endpoint=self.endpoint_name,
            action=action
        ).observe(duration)
        
        self.push_metrics()
    
    def track_inference(self, action: str, duration: float):
        """Track inference time"""
        inference_duration.labels(
            endpoint=self.endpoint_name,
            action=action
        ).observe(duration)
        
        self.push_metrics()
    
    def track_cold_start(self):
        """Track cold start"""
        if self.is_cold_start:
            cold_start_total.labels(endpoint=self.endpoint_name).inc()
            self.is_cold_start = False
            self.push_metrics()
    
    def track_embedding(self, dimension: int):
        """Track embedding dimension"""
        embedding_dimension.labels(endpoint=self.endpoint_name).set(dimension)
        self.push_metrics()
    
    def track_scores(self, scores: list):
        """Track cross-encoder scores"""
        if not scores:
            return
        
        for score in scores:
            score_value.labels(endpoint=self.endpoint_name).observe(score)
        
        score_min.labels(endpoint=self.endpoint_name).set(min(scores))
        score_max.labels(endpoint=self.endpoint_name).set(max(scores))
        
        self.push_metrics()
    
    def track_batch_size(self, size: int):
        """Track batch size"""
        batch_size.labels(endpoint=self.endpoint_name).observe(size)
        self.push_metrics()
    
    def track_cost(self, cost: float, request_count: int = 1):
        """Track cost metrics"""
        cost_per_request.labels(endpoint=self.endpoint_name).set(cost / request_count)
        cost_per_1k.labels(endpoint=self.endpoint_name).set(cost * 1000 / request_count)
        self.push_metrics()


# Decorator for tracking requests
def track_metrics(exporter: MetricsExporter, action: str):
    """Decorator to track request metrics"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                exporter.track_request(action, status, duration)
        
        return wrapper
    return decorator