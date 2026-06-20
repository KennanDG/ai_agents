import path from "node:path";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import electron from "vite-plugin-electron/simple";

const rootDirectory = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    electron({
      main: {
        entry: "electron/main.ts",
      },
      preload: {
        input: path.join(rootDirectory, "electron/preload.ts"),
      },
    }),
  ],
});
