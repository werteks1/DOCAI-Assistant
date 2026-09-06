import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Dev-прокси: фронт ходит по относительным путям, как в LAN-режиме.
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/settings": "http://127.0.0.1:8000",
    },
  },
});
