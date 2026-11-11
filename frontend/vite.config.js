import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy API calls to the FastAPI backend during development.
    // This avoids CORS issues when running `npm run dev`.
    proxy: {
      // Target 127.0.0.1 explicitly (not "localhost"). Node's DNS
      // resolution of "localhost" can prefer the IPv6 loopback (::1)
      // on this machine, while the FastAPI backend is started bound to
      // 127.0.0.1 (IPv4) only (see backend/main.py / uvicorn invocation).
      // That mismatch made Vite's proxy fail with ECONNREFUSED on ::1,
      // which the browser observed as generic 500 Internal Server Error
      // responses for /health and /api/* even though the backend itself
      // was fully healthy and reachable on 127.0.0.1:8000 directly.
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
