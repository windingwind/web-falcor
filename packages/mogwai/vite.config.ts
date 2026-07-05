import { defineConfig } from "vite";

export default defineConfig({
    server: {
        port: 5173,
        fs: {
            // Allow serving workspace packages and generated shader artifacts.
            allow: ["../.."],
        },
    },
});
