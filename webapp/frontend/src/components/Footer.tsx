export function Footer() {
  return (
    <footer class="footer">
      <div class="container footer__grid">
        <div class="footer__col">
          <span class="wordmark">VectoVecto</span>
          <p class="footer__note">
            Offline document restoration. Devanagari first, English too. Photo or scan in — cleaned
            page, searchable PDF, overlay, transcript and OCR JSON out, with every uncertain number
            on the record.
          </p>
        </div>

        <div class="footer__col">
          <span class="eyebrow">Read</span>
          <a class="footer__link" href="/#examples">
            Real examples
          </a>
          <a class="footer__link" href="/#pipeline">
            How it reads a page
          </a>
          <a class="footer__link" href="/#queue">
            The review queue
          </a>
          <a class="footer__link" href="/#evidence">
            Measured numbers
          </a>
        </div>

        <div class="footer__col">
          <span class="eyebrow">Project</span>
          <a
            class="footer__link"
            href="https://github.com/DontHash/VectoVecto"
            target="_blank"
            rel="noreferrer noopener"
          >
            GitHub — DontHash/VectoVecto
          </a>
          <span class="eyebrow">Docs</span>
          <span class="footer__link">docs/EVALUATION.md · docs/DEPLOY.md</span>
          <span class="footer__link">MIT · third-party models keep their licenses</span>
        </div>
      </div>

      <div class="container" style={{ "margin-top": "40px" }}>
        <hr class="rule" />
        <p class="eyebrow" style={{ "margin-top": "16px" }}>
          This demo deletes every upload after its retention window · the offline guarantee belongs
          to the local app
        </p>
      </div>
    </footer>
  );
}
