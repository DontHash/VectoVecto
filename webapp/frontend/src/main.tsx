import { render } from "solid-js/web";

import "./styles/fonts.css";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/components.css";
import "./styles/pages.css";

import { App } from "./App";

const root = document.getElementById("root");
if (!root) throw new Error("#root missing");

render(() => <App />, root);
