import { createReadStream, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve, sep } from "node:path";

import { type Connect, type Plugin, defineConfig } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));

/**
 * Local only (dev server and `vite preview`): serve a directory such as the
 * weights without copying it into the build.
 */
function serveDir(urlPrefix: string, dir: string): Plugin {
  const middleware: Connect.NextHandleFunction = (req, res, next) => {
    const path = resolve(dir, "." + decodeURIComponent((req.url ?? "/").split("?")[0]));
    if (!path.startsWith(dir + sep)) return next();
    let size: number;
    try {
      const st = statSync(path);
      if (!st.isFile()) return next();
      size = st.size;
    } catch {
      return next();
    }
    res.setHeader("Content-Length", size);
    res.setHeader("Content-Type", "application/octet-stream");
    res.setHeader("Cross-Origin-Resource-Policy", "same-origin");
    createReadStream(path).pipe(res);
  };
  return {
    name: `serve-dir:${urlPrefix}`,
    configureServer: (server) => void server.middlewares.use(urlPrefix, middleware),
    configurePreviewServer: (server) => void server.middlewares.use(urlPrefix, middleware),
  };
}

export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  plugins: [serveDir("/weights", resolve(root, "weights")), serveDir("/refs", resolve(root, "refs"))],
  build: {
    outDir: resolve(root, "dist"),
    emptyOutDir: true,
    target: "es2022",
  },
  server: {
    // Cross-origin isolation enables SharedArrayBuffer (multi-threaded wasm).
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  preview: { port: 4173 },
});
