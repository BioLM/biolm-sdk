/** Link sidebar section captions to their index pages (CLI, SDK). */
(function () {
  var CAPTION_TARGETS = {
    CLI: "cli/index.html",
    SDK: "sdk/index.html",
  };

  function linkCaptionTargets() {
    var root =
      typeof DOCUMENTATION_OPTIONS !== "undefined" && DOCUMENTATION_OPTIONS.URL_ROOT
        ? DOCUMENTATION_OPTIONS.URL_ROOT
        : "";
    document.querySelectorAll("p.caption .caption-text").forEach(function (span) {
      var label = span.textContent.trim();
      var target = CAPTION_TARGETS[label];
      if (!target || span.closest("a.caption-link")) {
        return;
      }
      var caption = span.closest("p.caption");
      if (!caption) {
        return;
      }
      var link = document.createElement("a");
      link.href = root + target;
      link.className = "caption-link";
      caption.insertBefore(link, span);
      link.appendChild(span);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", linkCaptionTargets);
  } else {
    linkCaptionTargets();
  }
})();
