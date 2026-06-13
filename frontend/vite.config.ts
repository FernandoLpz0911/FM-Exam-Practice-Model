import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,   // FM frontend on 5174; P engine on 5173
    proxy: {
      '/api': {
        target: 'http://localhost:8001',   // FM engine on 8001
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
