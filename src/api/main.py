"""FastAPI Main Application for OceanEmbed Production Backend.

Provides RESTful API for:
- 3D ocean temperature reconstructions and calibrated uncertainty
- Downstream disaster management products (TCHP, OHC, MHW, MLD)
- Domain metadata and healthchecks
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes_products import router as products_router

app = FastAPI(
    title="OceanEmbed Operational Disaster API",
    description="Backend inference and disaster product service for OceanEmbed (SIH PS26066)",
    version="1.0.0",
)

# Enable CORS for Next.js frontend (localhost:3000) and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include disaster products router
app.include_router(products_router)


@app.get("/")
def root_endpoint():
    return {
        "service": "OceanEmbed Disaster Products API",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": [
            "/products/profile",
            "/products/tchp",
            "/products/heatwave_status",
            "/products/heatwave_grid",
            "/products/domain_info",
        ],
    }


@app.get("/health")
def healthcheck():
    return {"status": "healthy", "service": "oceanembed-backend"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
