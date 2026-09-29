import { A, useLocation } from "@solidjs/router";

export function TopBar() {
  const location = useLocation();
  const isStudio = () => location.pathname.startsWith("/studio");

  return (
    <header class="topbar">
      <div class="container topbar__inner">
        <A href="/" class="wordmark" aria-label="VectoVecto home">
          Vecto<span>Vecto</span>
        </A>

        <nav class="topbar__nav" aria-label="Primary">
          <a class="nav-link" href="/#examples">
            Examples
          </a>
          <a class="nav-link" href="/#pipeline">
            How it reads
          </a>
          <a class="nav-link" href="/#queue">
            Review queue
          </a>
          <a class="nav-link" href="/#evidence">
            Evidence
          </a>
        </nav>

        <div class="topbar__actions">
          <a
            class="nav-link topbar__gh"
            href="https://github.com/DontHash/VectoVecto"
            rel="noreferrer noopener"
            target="_blank"
          >
            GitHub
          </a>
          <A class="btn btn--ghost btn--small" href="/studio" data-active={isStudio()}>
            Open the studio
          </A>
        </div>
      </div>
    </header>
  );
}
