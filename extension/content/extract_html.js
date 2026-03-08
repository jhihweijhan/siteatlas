(function () {
  try {
    return {
      html: document.documentElement.outerHTML,
      url: location.href,
      title: document.title
    };
  } catch (e) {
    return {
      error: `Failed to extract HTML: ${e.message}`,
      url: location.href,
      title: document.title
    };
  }
})();
