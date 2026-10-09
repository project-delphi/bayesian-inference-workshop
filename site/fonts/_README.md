# Fonts

Two variable fonts, vendored so the site does not depend on a font CDN. They are the
same subset files as `project-delphi/tensors-workshop/fonts/`, which documents how they
were fetched from `google/fonts` and subset with fontTools. The subset covers Latin,
Greek, arrows, mathematical operators, letterlike symbols and general punctuation, so
prose such as "θ → η" stays in the text face.

| File | Family | Use | SHA-256 |
|---|---|---|---|
| `inter-latin.woff2` | Inter (`wght` 400–800) | body text, UI | `a7011f8369ba7d032805f89080c9677746a9dc115e9d457f4c8bdac2e8ef006f` |
| `source-serif-4-latin.woff2` | Source Serif 4 (`opsz`, `wght` 200–900) | headings | `c2a5c004a5e5fd24d54169e3f9a544ea84584dcf0d66c6c2bcac6db0219c70ef` |

Both are licensed under the SIL Open Font License 1.1; the licence texts are
`OFL-Inter.txt` and `OFL-SourceSerif4.txt`, copied unmodified from `google/fonts`.

`fonts.css` holds the `@font-face` rules; `_quarto.yml` links it under `format.html.css`
and lists `fonts/**` as a project resource. Verify with:

```bash
shasum -a 256 site/fonts/*.woff2
```
