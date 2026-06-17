/**
 * Instant replies for greetings and other messages that need no DB/RAG lookup.
 * Keep in sync with is_small_talk() / small_talk_response() in unified_chatbot.py.
 */
(function initInstantReplies(global) {
  const EXACT = new Set([
    'hi', 'hello', 'hey', 'yo', 'hiya', 'howdy', 'sup',
    'thanks', 'thank you', 'thx', 'ty',
    'bye', 'goodbye', 'cya', 'see ya', 'see you',
    'ok', 'okay', 'cool', 'great', 'help',
  ]);

  const PATTERNS = [
    /^(hi|hello|hey|yo|hiya|howdy)\s*(there|everyone|friend)?$/,
    /^how are you$/,
    /^how r u$/,
    /^how('re| is) you$/,
    /^how('s| is) it going$/,
    /^what('s| is) up$/,
    /^good (morning|afternoon|evening|night)$/,
    /^who are you$/,
    /^what are you$/,
    /^what can you (do|help with)$/,
    /^how does this work$/,
    /^how do i use this$/,
    /^what is this$/,
    /^what do you do$/,
  ];

  const DOMAIN_HINTS = [
    'event', '311', '911', 'crime', 'safety', 'dorchester', 'news',
    'meeting', 'happening', 'request', 'neighborhood', 'budget', 'policy',
    'arrest', 'shooting', 'calendar', 'schedule', 'activity', 'service',
  ];

  function normalize(message) {
    return String(message || '')
      .trim()
      .toLowerCase()
      .replace(/[^\w\s']/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function isInstant(message) {
    const raw = String(message || '').trim();
    if (!raw || raw.length > 120) return false;
    const lower = raw.toLowerCase();
    if (DOMAIN_HINTS.some((hint) => lower.includes(hint))) return false;
    const n = normalize(raw);
    if (!n) return false;
    if (EXACT.has(n)) return true;
    return PATTERNS.some((pat) => pat.test(n));
  }

  function reply(message) {
    const n = normalize(message);
    if (n.includes('how are you') || n.includes('how r u') || n.includes('how is it going') || n.includes("how's it going")) {
      return (
        "I'm doing well — thanks for asking! I'm here to help with Dorchester community "
        + 'info: events, 311 activity, safety trends, and neighborhood news. '
        + 'What would you like to know?'
      );
    }
    if (n.startsWith('thanks') || n.startsWith('thank you') || n.startsWith('thx') || n === 'ty') {
      return (
        "You're welcome! Ask anytime about events, city services, safety, or what's "
        + 'happening in the neighborhood.'
      );
    }
    if (n.startsWith('bye') || n.startsWith('goodbye') || n.startsWith('cya') || n.startsWith('see ya') || n.startsWith('see you')) {
      return 'Goodbye! Come back anytime you have questions about Dorchester.';
    }
    if (n.includes('who are you') || n.includes('what are you') || n.includes('what can you') || n.includes('what do you do')) {
      return (
        "I'm your Dorchester community assistant. I can help with local events, 311 requests, "
        + 'safety data, and neighborhood news. Try “Events this week” or “311 activity” to get started.'
      );
    }
    if (n.includes('how does this work') || n.includes('how do i use this') || n === 'what is this' || n === 'help') {
      return (
        'I can answer questions about Dorchester events, 311 service requests, safety trends, '
        + 'and community news. Try one of the suggestion chips below, or ask in your own words.'
      );
    }
    if (n.startsWith('good ')) {
      return (
        "Good to see you! Ask me about Dorchester events, services, safety, or neighborhood trends — "
        + "what's on your mind?"
      );
    }
    return (
      "Hi! I'm here to help with Dorchester community questions — events, services, safety, "
      + 'and local news. What would you like to know?'
    );
  }

  global.InstantReplies = { isInstant, reply, normalize };
})(window);
