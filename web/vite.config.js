import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [react()],
  publicDir: resolve(__dirname, "../outputs"),
  server: {
    port: 5173,
    strictPort: false,
  },
  preview: {
    port: 4173,
    strictPort: false,
  },
});
