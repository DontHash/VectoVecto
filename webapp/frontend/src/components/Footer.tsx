export function Footer() {
  return (
    <footer class="footer">
      <div class="container footer__grid">
        <div class="footer__col">
          <span class="wordmark">VeriScript</span>
          <p class="footer__note">
            Offline document restoration. Devanagari first, English too. Photo or scan in — cleaned
            page, searchable PDF, overlay, transcript and OCR JSON out, with every uncertain number
            on the record.
          </p>
        </div>

        <div class="footer__col">
          <span class="eyebrow">Explore</span>
          <a class="footer__link" href="/#examples">
            Real examples
          </a>
          <a class="footer__link" href="/#offline">
            Runs offline
          </a>
          <a class="footer__link" href="/studio">
            Try the studio
          </a>
          <a class="footer__link" href="/nepali-ocr/">
            Nepali OCR — offline
          </a>
          <a
            class="footer__link"
            href="https://github.com/DontHash/VeriScript/blob/main/docs/BENCHMARK.md"
            target="_blank"
            rel="noreferrer noopener"
          >
            Benchmark — Devanagari head-to-head
          </a>
        </div>

        <div class="footer__col">
          <span class="eyebrow">Project</span>
          <a
            class="footer__link"
            href="https://github.com/DontHash/VeriScript"
            target="_blank"
            rel="noreferrer noopener"
          >
            GitHub — DontHash/VeriScript
          </a>
          <span class="eyebrow">Docs</span>
          <span class="footer__link">docs/BENCHMARK.md · docs/EVALUATION.md · docs/DEPLOY.md</span>
          <span class="footer__link">MIT · third-party models keep their licenses</span>
        </div>
      </div>

      <div class="container footer__meta">
        <hr class="rule" />
        <p class="eyebrow footer__note footer__note--top">
          This demo deletes every upload after its retention window · the offline guarantee belongs
          to the local app
        </p>
      </div>
    </footer>
  );
}
