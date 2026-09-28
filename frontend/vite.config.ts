import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Dev-only convenience: the app calls relative /api/* paths, this
      // forwards them to the FastAPI dev server so there's no CORS
      // config to maintain for local development. Production build uses
      // VITE_API_URL directly (see src/api/client.ts).
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
