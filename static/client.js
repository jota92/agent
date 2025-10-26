(() => {
  const canvas = document.getElementById("browser-canvas");
  const statusLabel = document.getElementById("status-text");
  const ctx = canvas.getContext("2d");
  const socket = io();

  const activeKeys = new Map();
  const deviceScale = window.devicePixelRatio || 1;
  let inputActive = false;
  let lastFrameToken = 0;

  function setStatus(message) {
    statusLabel.textContent = message;
  }

  function emitViewport() {
    const width = Math.floor(window.innerWidth * deviceScale);
    const height = Math.floor(window.innerHeight * deviceScale);
    socket.emit("viewport", { width, height });
  }

  function emitMouseEvent(type, event) {
    if (!socket.connected) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    socket.emit("mouse-event", {
      type,
      x,
      y,
      button: event.button,
      deltaX: (event.deltaX || 0) * deviceScale,
      deltaY: (event.deltaY || 0) * deviceScale,
    });
  }

  function activateInput() {
    if (!inputActive) {
      inputActive = true;
      setStatus("Chromiumに接続しました - 入力を送信中");
    }
  }

  function deactivateInput() {
    if (inputActive) {
      inputActive = false;
      setStatus("クリックしてChromiumを操作してください。");
    }
  }

  socket.on("connect", () => {
    setStatus("接続済み - クリックで操作を開始");
    emitViewport();
  });

  socket.on("disconnect", () => {
    deactivateInput();
    setStatus("切断されました。リロードしてください。");
  });

  socket.on("status", (payload) => {
    if (payload && payload.message) {
      setStatus(payload.message);
    }
  });

  socket.on("frame", (payload) => {
    const data = payload && payload.data;
    if (!data) {
      return;
    }
    const token = ++lastFrameToken;
    const image = new Image();
    image.onload = () => {
      if (token !== lastFrameToken) {
        return;
      }
      canvas.width = image.width;
      canvas.height = image.height;
      canvas.style.width = `${image.width / deviceScale}px`;
      canvas.style.height = `${image.height / deviceScale}px`;
      ctx.drawImage(image, 0, 0);
    };
    image.src = `data:image/jpeg;base64,${data}`;
  });

  canvas.setAttribute("tabindex", "0");

  canvas.addEventListener("mousedown", (event) => {
    event.preventDefault();
    canvas.focus({ preventScroll: true });
    activateInput();
    emitMouseEvent("down", event);
  });

  canvas.addEventListener("mouseup", (event) => {
    event.preventDefault();
    emitMouseEvent("up", event);
  });

  canvas.addEventListener("mousemove", (event) => {
    event.preventDefault();
    emitMouseEvent("move", event);
  });

  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      emitMouseEvent("wheel", event);
    },
    { passive: false },
  );

  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });

  document.addEventListener("keydown", (event) => {
    if (!inputActive || !socket.connected) {
      return;
    }
    event.preventDefault();
    activeKeys.set(event.code, event.key);
    socket.emit("keyboard-event", {
      type: "keydown",
      key: event.key,
      code: event.code,
      repeat: event.repeat,
    });
  });

  document.addEventListener("keyup", (event) => {
    if (!inputActive || !socket.connected) {
      return;
    }
    event.preventDefault();
    activeKeys.delete(event.code);
    socket.emit("keyboard-event", {
      type: "keyup",
      key: event.key,
      code: event.code,
    });
  });

  window.addEventListener("blur", () => {
    if (socket.connected) {
      for (const [code, key] of activeKeys.entries()) {
        socket.emit("keyboard-event", { type: "keyup", key, code });
      }
    }
    activeKeys.clear();
    deactivateInput();
  });

  window.addEventListener("resize", () => {
    emitViewport();
  });

  deactivateInput();
})();
