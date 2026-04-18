from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request,HTTPException
from fastapi.responses import JSONResponse

csp_policy = (
            "default-src 'self'; "
            "script-src 'self' https: 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' https: 'unsafe-inline' 'unsafe-eval'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'none'; "
            "block-all-mixed-content; "
            "upgrade-insecure-requests; "
            "img-src 'self' https: data:; "
            "font-src 'self' https: data:; "
            "style-src 'self' https: 'unsafe-inline'; "
            "connect-src 'self';"
        )


    

    
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers['Server'] = 'Frontend'  
        response.headers['X-Powered-By'] = ''
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers["Cache-Control"] = "no-store"
        response.headers['Expect-CT'] = 'max-age=86400, enforce'
        response.headers['Feature-Policy'] = "geolocation 'none'; microphone 'none'"
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers["Clear-Site-Data"] = '"cache", "cookies", "storage", "executionContexts"'
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        response.headers["Content-Security-Policy"] = csp_policy
        response.headers["NEL"] = '{"report_to": "default", "max_age": 31536000, "include_subdomains": true}'
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        return response
    
class BlockReDocMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/redoc":
            return JSONResponse(status_code=403, content={"detail": "Access forbidden"})
        return await call_next(request)

class RestrictSwaggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/docs"):
            # Replace this condition with your own IP restriction or other logic
            return JSONResponse({"detail": "Access to Swagger UI is restricted"}, status_code=403)
        response = await call_next(request)
        return response


def check_no_query_params(request: Request):
    if request.query_params:
        raise HTTPException(
            status_code=400,
            detail="Query parameters are not allowed."
        )
    return request