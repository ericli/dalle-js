/**
 * Port of min-dalle's TextTokenizer (min_dalle/text_tokenizer.py).
 *
 * This is not byte-level BPE: the text is lowercased, stripped to ASCII, split
 * on single spaces, and each word is merged from characters with a leading
 * "Ġ". Known gap: min-dalle first runs emoji.demojize (🐶 -> "dog_face"); we
 * skip that, so emoji are dropped by the ASCII filter instead.
 */
export class TextTokenizer {
  private readonly vocab: Map<string, number>;
  private readonly subwords: string[] = [];
  private readonly ranks: Map<string, number>;
  private readonly cls: number;
  private readonly sep: number;
  private readonly unk: number;

  constructor(vocab: Record<string, number>, mergesText: string) {
    this.vocab = new Map(Object.entries(vocab));
    for (const [subword, id] of this.vocab) this.subwords[id] = subword;
    // Python: f.read().split("\n")[1:-1] drops the "#version" header line and
    // whatever follows the last newline. Later duplicates overwrite earlier.
    const merges = mergesText.split("\n").slice(1, -1);
    this.ranks = new Map();
    merges.forEach((line, rank) => {
      const [a, b] = line.trim().split(/\s+/);
      this.ranks.set(pairKey(a, b), rank);
    });
    this.cls = this.lookup("<s>");
    this.sep = this.lookup("</s>");
    this.unk = this.lookup("<unk>");
  }

  private lookup(token: string): number {
    const id = this.vocab.get(token);
    if (id === undefined) throw new Error(`token ${token} missing from vocab`);
    return id;
  }

  /** The subword text of a token id, with the word-start marker "Ġ" as a space. */
  subword(id: number): string {
    return (this.subwords[id] ?? "").replace(/^\u0120/, " ");
  }

  tokenize(text: string): number[] {
    // Lowercase, then drop non-ASCII (Python: .lower().encode("ascii", "ignore")).
    text = text.toLowerCase().replace(/[^\x00-\x7f]/g, "");
    const tokens = [this.cls];
    for (const word of text.split(" ")) {
      if (word.length === 0) continue;
      for (const subword of this.bytePairEncoding(word)) {
        tokens.push(this.vocab.get(subword) ?? this.unk);
      }
    }
    tokens.push(this.sep);
    return tokens;
  }

  private bytePairEncoding(word: string): string[] {
    let subwords = ["Ġ", ...word]; // chr(ord(" ") + 256) == "Ġ"
    while (subwords.length > 1) {
      // Lowest-ranked adjacent pair; the first one wins ties, like Python's min().
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < subwords.length - 1; i++) {
        const rank = this.ranks.get(pairKey(subwords[i], subwords[i + 1]));
        if (rank !== undefined && rank < bestRank) {
          best = i;
          bestRank = rank;
        }
      }
      if (best < 0) break;
      subwords = [
        ...subwords.slice(0, best),
        subwords[best] + subwords[best + 1],
        ...subwords.slice(best + 2),
      ];
    }
    return subwords;
  }
}

function pairKey(a: string, b: string): string {
  return a + "\u0000" + b;
}

/**
 * Build the [2, 64] text token rows min-dalle feeds the encoder: row 0 is the
 * unconditioned prompt (just BOS and the last token), row 1 the full prompt,
 * both padded with 1 (<pad>).
 */
export function textTokenRows(tokens: number[], textTokenCount = 64): Int32Array<ArrayBuffer> {
  tokens = tokens.slice(0, textTokenCount);
  const rows = new Int32Array(2 * textTokenCount).fill(1);
  rows[0] = tokens[0];
  rows[1] = tokens[tokens.length - 1];
  rows.set(tokens, textTokenCount);
  return rows;
}
