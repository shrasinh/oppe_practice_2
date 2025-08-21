from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import logging
import time
import json
import joblib  
import pandas as pd  

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Setup Tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Setup structured logging
logger = logging.getLogger("mlops-assignment-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

# create a custom formatter class
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record)
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key in ["event", "trace_id", "path", "error", "latency_ms"]:
                log_obj[key] = value
                
        return json.dumps(log_obj, default=str)

handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

# FastAPI app
app = FastAPI(title="Regression")

model = None

class Schema(BaseModel):
    longitude: float
    latitude: float 
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float 
    median_income: float
    high_income: float
    

app_state = {"is_ready": False, "is_alive": True}

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = joblib.load("artifacts/california_housing_model.joblib")
        app_state["is_ready"] = True
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        app_state["is_ready"] = False
        raise e

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    
    logger.exception(
        "Unhandled exception occurred",
        extra={
            "event": "unhandled_exception",
            "trace_id": trace_id,
            "path": str(request.url),
            "error": str(exc)
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

@app.get("/")
def home():
    return {"message": "mlops assignment"}

@app.post("/predict")
async def predict(data: Schema, request: Request):
    if not app_state["is_ready"] or model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")
        
        try:
            input_data = pd.DataFrame([data.dict()])
            output = model.predict(input_data)[0]
            latency = round((time.time() - start_time) * 1000, 2)
            
            # Log successful prediction
            logger.info(
                "Prediction completed successfully",
                extra={
                    "event": "prediction_success",
                    "trace_id": trace_id,
                    "latency_ms": latency
                }
            )
            
            return {"predicted": output}
            
        except Exception as e:
            logger.exception(
                "Prediction failed",
                extra={
                    "event": "prediction_error",
                    "trace_id": trace_id,
                    "error": str(e)
                }
            )
            raise HTTPException(status_code=500, detail="Prediction failed")