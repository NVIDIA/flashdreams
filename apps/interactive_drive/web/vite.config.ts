import { defineConfig } from 'vite';
import driveExportPlugin from './export-api.mjs';

export default defineConfig({
  plugins: [driveExportPlugin()],
  build: { rollupOptions: { input: { cockpit: 'index.html', comparison: 'comparison.html' } } },
  preview: { host: '127.0.0.1' },
  server: {
    host: '0.0.0.0',
    port: 5173,
    watch: { usePolling: true, interval: 500 },
    proxy: {
      '/api/webrtc': {
        target: process.env.VITE_BACKEND_URL || 'http://127.0.0.1:8089',
        changeOrigin: true,
      },
    },
  },
});
