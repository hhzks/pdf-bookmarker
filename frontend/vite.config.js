import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // Local dev: the FastAPI backend runs on :8000 and VITE_API_BASE_URL is unset.
    proxy: { "/api": "http://localhost:8000" },
  },
});
