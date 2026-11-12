import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// Keep capture, pose and inference requests on the same backend instance.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendTarget = env.VITE_PROXY_TARGET || 'http://127.0.0.1:18901'

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/api': { target: backendTarget, changeOrigin: true },
        '/health': { target: backendTarget, changeOrigin: true },
      },
    },
  }
})
