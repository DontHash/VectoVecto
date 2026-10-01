import { render } from "solid-js/web";

import "./styles/fonts.css";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/components.css";
import "./styles/pages.css";

import { App } from "./App";

const root = document.getElementById("root");
if (!root) throw new Error("#root missing");

// Remove the crawlable static fallback from index.html before mounting.
// Deliberately a bundled module, not an inline script: the production CSP is
// `script-src 'self'`, so inline removal is blocked (regression: the fallback
// stayed in the DOM and rendered above the app).
document.getElementById("seo-fallback")?.remove();

render(() => <App />, root);
