import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // В dev браузер бьёт в тот же origin (5173…), Vite пересылает на AI-сервис — меньше сбоев, чем прямой localhost:8000.
    proxy: {
      "/analyze": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/health": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
