/**
 * Code Snippets Skill for OpenClaw
 * Session 8 Hands-On: Cut the Crap — AI Engineer Edition
 *
 * Save, search, and manage code snippets through your AI assistant.
 */

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

let snippets = [];
let storagePath = "./snippets.json";

// --- Persistence ---

function loadSnippets() {
  try {
    if (fs.existsSync(storagePath)) {
      const data = fs.readFileSync(storagePath, "utf-8");
      snippets = JSON.parse(data);
      console.log(`Loaded ${snippets.length} snippets from ${storagePath}`);
    }
  } catch (err) {
    console.error(`Failed to load snippets: ${err.message}`);
    snippets = [];
  }
}

function saveSnippets() {
  try {
    fs.writeFileSync(storagePath, JSON.stringify(snippets, null, 2));
  } catch (err) {
    console.error(`Failed to save snippets: ${err.message}`);
  }
}

// --- Skill Lifecycle ---

module.exports = {
  async onLoad(context) {
    if (context.config && context.config.STORAGE_PATH) {
      storagePath = context.config.STORAGE_PATH;
    }
    loadSnippets();
    console.log("✅ Code Snippets skill loaded");
  },

  // --- Tool Handlers ---
  tools: {
    /**
     * Save a new code snippet.
     */
    async save_snippet({ title, code, language, tags = [] }, context) {
      const id = crypto.randomBytes(4).toString("hex");
      const snippet = {
        id,
        title,
        code,
        language: language.toLowerCase(),
        tags: tags.map((t) => t.toLowerCase()),
        createdAt: new Date().toISOString(),
      };

      snippets.push(snippet);
      saveSnippets();

      return {
        message: `Snippet saved successfully!`,
        id: snippet.id,
        title: snippet.title,
        language: snippet.language,
        tags: snippet.tags,
      };
    },

    /**
     * Search snippets by keyword, language, or tag.
     */
    async search_snippets({ query, language, tag }, context) {
      let results = [...snippets];

      // Filter by language
      if (language) {
        results = results.filter(
          (s) => s.language === language.toLowerCase()
        );
      }

      // Filter by tag
      if (tag) {
        results = results.filter((s) =>
          s.tags.includes(tag.toLowerCase())
        );
      }

      // Search by keyword in title, code, and tags
      if (query) {
        const q = query.toLowerCase();
        results = results.filter(
          (s) =>
            s.title.toLowerCase().includes(q) ||
            s.code.toLowerCase().includes(q) ||
            s.tags.some((t) => t.includes(q))
        );
      }

      if (results.length === 0) {
        return { message: "No snippets found matching your criteria.", results: [] };
      }

      return {
        message: `Found ${results.length} snippet(s)`,
        results: results.map((s) => ({
          id: s.id,
          title: s.title,
          language: s.language,
          tags: s.tags,
          code: s.code,
          createdAt: s.createdAt,
        })),
      };
    },

    /**
     * List recent snippets.
     */
    async list_snippets({ limit = 10, language }, context) {
      let results = [...snippets];

      if (language) {
        results = results.filter(
          (s) => s.language === language.toLowerCase()
        );
      }

      // Sort by newest first, take limit
      results = results
        .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
        .slice(0, limit);

      if (results.length === 0) {
        return { message: "No snippets saved yet.", results: [] };
      }

      return {
        message: `Showing ${results.length} snippet(s)`,
        total: snippets.length,
        results: results.map((s) => ({
          id: s.id,
          title: s.title,
          language: s.language,
          tags: s.tags,
          preview: s.code.substring(0, 100) + (s.code.length > 100 ? "..." : ""),
          createdAt: s.createdAt,
        })),
      };
    },

    /**
     * Delete a snippet by ID.
     */
    async delete_snippet({ id }, context) {
      const index = snippets.findIndex((s) => s.id === id);

      if (index === -1) {
        return { error: `Snippet with ID '${id}' not found.` };
      }

      const removed = snippets.splice(index, 1)[0];
      saveSnippets();

      return {
        message: `Deleted snippet '${removed.title}'`,
        id: removed.id,
      };
    },
  },
};
