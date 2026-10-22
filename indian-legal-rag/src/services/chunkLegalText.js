const crypto = require("node:crypto");

function splitLegalText(text, maxChars = 3000) {
  // Preserve paragraph and clause boundaries where possible.
  const paragraphs = text
    .split(/\n\s*\n/)
    .map(p => p.trim())
    .filter(Boolean);

  const chunks = [];
  let current = "";

  for (const paragraph of paragraphs) {
    if (paragraph.length > maxChars) {
      if (current) {
        chunks.push(current);
        current = "";
      }

      // Fallback for unusually long individual paragraphs.
      // Prefer clause-aware splitting for these records.
      for (let i = 0; i < paragraph.length; i += maxChars) {
        chunks.push(paragraph.slice(i, i + maxChars));
      }
      continue;
    }

    if (
      current &&
      current.length + paragraph.length + 2 > maxChars
    ) {
      chunks.push(current);
      current = paragraph;
    } else {
      current = current
        ? `${current}\n\n${paragraph}`
        : paragraph;
    }
  }

  if (current) chunks.push(current);

  return chunks;
}

function chunkLegalRecord(record) {
  if (!record.text || !record.text.trim()) {
    return [];
  }

  const documentId =
    record.document_id || record.id || "unknown_document";

  const baseText = [
    record.title,
    record.document_type,
    record.article_or_section
      ? `Article/Section: ${record.article_or_section}`
      : null,
    record.heading,
    record.text
  ].filter(Boolean).join("\n");

  return splitLegalText(record.text).map((text, index) => {
    const chunkId = crypto
      .createHash("sha256")
      .update(`${documentId}:${index}:${text}`)
      .digest("hex");

    return {
      chunk_id: chunkId,
      document_id: documentId,
      document_type: record.document_type || "UNKNOWN",
      title: record.title || "",
      article_or_section: record.article_or_section || "",
      heading: record.heading || "",
      text,
      embedding_text: [
        record.title,
        record.document_type,
        record.article_or_section
          ? `Article/Section: ${record.article_or_section}`
          : null,
        record.heading,
        text
      ].filter(Boolean).join("\n"),
      source_url: record.source_url || "",
      version: record.version || "",
      effective_from: record.effective_from ?? null,
      effective_to: record.effective_to ?? null,
      language: record.language || "en",
      source_record_id: documentId,
      chunk_index: index
    };
  });
}

module.exports = { chunkLegalRecord };
