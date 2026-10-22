const fs = require("node:fs");
const path = require("node:path");
const {
  chunkLegalRecord
} = require("../src/services/chunkLegalText");

const inputFiles = [
  "constitution.cleaned.json",
  "ipc.cleaned.json",
  "crpc.cleaned.json"
];

const outputFile = path.join(
  __dirname,
  "../data/chunks/legal-chunks.json"
);

function loadRecords(filePath) {
  const data = JSON.parse(
    fs.readFileSync(filePath, "utf8")
  );

  // Supports either a top-level array or { records: [...] }.
  if (Array.isArray(data)) return data;
  if (Array.isArray(data.records)) return data.records;

  throw new Error(
    `${filePath} must contain an array of records`
  );
}

function main() {
  const allChunks = [];

  for (const filename of inputFiles) {
    const filePath = path.join(
      __dirname,
      "../data/cleaned",
      filename
    );

    if (!fs.existsSync(filePath)) {
      throw new Error(`Missing input file: ${filePath}`);
    }

    const records = loadRecords(filePath);

    for (const record of records) {
      allChunks.push(...chunkLegalRecord(record));
    }

    console.log(
      `${filename}: ${records.length} records processed`
    );
  }

  fs.mkdirSync(path.dirname(outputFile), {
    recursive: true
  });

  fs.writeFileSync(
    outputFile,
    JSON.stringify(allChunks, null, 2)
  );

  console.log(`Total chunks: ${allChunks.length}`);
  console.log(`Saved to: ${outputFile}`);
}

main();