/* theme-sync.js — reads the saved theme from localStorage and applies it to
 * blog article pages. Loaded as the first script in <head> so the attribute
 * is present before any CSS cascade is evaluated, preventing a flash of the
 * wrong theme. Also listens for storage events so the article updates in real
 * time when the user toggles the theme on the main page in another tab. */
(function () {
  function applyTheme(t) {
    if (t === 'paper-dark') {
      document.documentElement.setAttribute('data-theme', 'paper-dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
  }

  try { applyTheme(localStorage.getItem('theme')); } catch (_) {}

  window.addEventListener('storage', function (e) {
    if (e.key === 'theme') {
      try { applyTheme(e.newValue); } catch (_) {}
    }
  });
})();
