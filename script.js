/* script.js — theme switcher, sticky-nav active-section state,
 * mobile nav toggle, and Formspree-aware contact form handler. */

(function () {
  'use strict';

  // ---------- Theme switcher (light / dark) ----------
  const THEMES = ['paper', 'paper-dark'];

  const root = document.documentElement;
  const btn = document.getElementById('theme-switch');

  const saved = (function () {
    try { return localStorage.getItem('theme'); } catch (_) { return null; }
  })();

  const initial = THEMES.includes(saved) ? saved : 'paper';
  root.dataset.theme = initial;
  if (btn) {
    const isDarkInitial = initial === 'paper-dark';
    btn.setAttribute('aria-pressed', String(isDarkInitial));
    btn.setAttribute('aria-label', isDarkInitial ? 'Switch to light theme' : 'Switch to dark theme');
  }

  if (btn) {
    btn.addEventListener('click', () => {
      const cur = root.dataset.theme || 'paper';
      const next = cur === 'paper' ? 'paper-dark' : 'paper';
      root.dataset.theme = next;
      const isDark = next === 'paper-dark';
      btn.setAttribute('aria-pressed', String(isDark));
      btn.setAttribute('aria-label', isDark ? 'Switch to light theme' : 'Switch to dark theme');
      try { localStorage.setItem('theme', next); } catch (_) {}
    });
  }

  // ---------- Mobile nav toggle ----------
  const navToggle = document.querySelector('.nav-toggle');
  const navLinks = document.getElementById('nav-links');
  const navSocial = document.getElementById('nav-social');

  if (navToggle && navLinks && navSocial) {
    navToggle.addEventListener('click', () => {
      const open = navLinks.classList.toggle('is-open');
      navSocial.classList.toggle('is-open', open);
      navToggle.setAttribute('aria-expanded', String(open));
    });

    // Collapse on link tap (mobile)
    navLinks.addEventListener('click', (e) => {
      if (e.target.tagName === 'A' && window.matchMedia('(max-width: 768px)').matches) {
        navLinks.classList.remove('is-open');
        navSocial.classList.remove('is-open');
        navToggle.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // ---------- Active section in nav ----------
  const sections = Array.from(document.querySelectorAll('main .section[id]'));
  const linksByHash = new Map();
  document.querySelectorAll('.nav-links a[href^="#"]').forEach((a) => {
    linksByHash.set(a.getAttribute('href'), a);
  });

  if (sections.length && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const id = entry.target.id;
          linksByHash.forEach((link) => link.classList.remove('is-active'));
          const active = linksByHash.get(`#${id}`);
          if (active) active.classList.add('is-active');
        });
      },
      { rootMargin: '-40% 0px -55% 0px', threshold: 0 }
    );
    sections.forEach((s) => io.observe(s));
  }

  // ---------- Contact form ----------
  const form = document.getElementById('contact-form');
  const status = document.getElementById('form-status');

  function setStatus(msg, state) {
    if (!status) return;
    status.textContent = msg;
    if (state) status.dataset.state = state; else status.removeAttribute('data-state');
  }

  if (form) {
    form.addEventListener('submit', async (e) => {
      // If the form action still has the placeholder, fall back to mailto so
      // the page is functional before Formspree is wired up.
      const action = form.getAttribute('action') || '';
      if (action.includes('YOUR_FORM_ID')) {
        e.preventDefault();
        setStatus('Form not yet wired — replace YOUR_FORM_ID in index.html with your Formspree endpoint.', 'error');
        return;
      }

      e.preventDefault();
      const submitBtn = form.querySelector('button[type="submit"]');
      const originalLabel = submitBtn ? submitBtn.textContent : '';
      if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = 'Sending…'; }
      setStatus('', null);

      try {
        const res = await fetch(action, {
          method: 'POST',
          body: new FormData(form),
          headers: { Accept: 'application/json' },
        });

        if (res.ok) {
          form.reset();
          setStatus('Thanks — your message is on its way.', 'ok');
        } else {
          const data = await res.json().catch(() => ({}));
          const errs = data && data.errors ? data.errors.map((x) => x.message).join(', ') : '';
          setStatus(errs || 'Something went wrong — please email me directly instead.', 'error');
        }
      } catch (_) {
        setStatus('Network error — please try again or email me directly.', 'error');
      } finally {
        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = originalLabel; }
      }
    });
  }
})();
