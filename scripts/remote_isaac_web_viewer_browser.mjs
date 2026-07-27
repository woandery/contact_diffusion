import { chromium } from "../outputs/remote_visualization/web-viewer-sample/node_modules/playwright/index.mjs";

const viewerUrl = process.env.ISAAC_WEB_VIEWER_URL ?? "http://127.0.0.1:5173";
const staticPageUrl = process.env.STATIC_PAGE_URL;
const chromiumPath = process.env.CHROMIUM_PATH;

if (!chromiumPath) {
  throw new Error("CHROMIUM_PATH must point to the Chromium executable");
}

const browser = await chromium.launch({
  executablePath: chromiumPath,
  headless: false,
  args: [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--autoplay-policy=no-user-gesture-required",
    "--start-maximized",
    "--window-size=1920,1080",
  ],
});

const context = await browser.newContext({
  viewport: null,
  ignoreHTTPSErrors: true,
});
const page = await context.newPage();
await page.addInitScript(() => {
  // The NVIDIA 4.4.2 viewer can clear an old React media node's srcObject
  // immediately after play() resolves, then dereference that null srcObject in
  // its promise callback. Keep the last live MediaStream visible to the old
  // node during the hand-off so the library can finish registering its tracks.
  const srcObjectDescriptor = Object.getOwnPropertyDescriptor(
    HTMLMediaElement.prototype,
    "srcObject",
  );
  const retainedStreams = new WeakMap();
  if (srcObjectDescriptor?.get && srcObjectDescriptor?.set) {
    Object.defineProperty(HTMLMediaElement.prototype, "srcObject", {
      configurable: srcObjectDescriptor.configurable,
      enumerable: srcObjectDescriptor.enumerable,
      get() {
        return srcObjectDescriptor.get.call(this) ?? retainedStreams.get(this) ?? null;
      },
      set(value) {
        if (value instanceof MediaStream) {
          retainedStreams.set(this, value);
        }
        // Do not expose a transient null while the NVIDIA viewer is awaiting
        // play(). The retained stream naturally becomes unreachable with the
        // discarded DOM node.
        if (value === null && retainedStreams.has(this)) {
          return;
        }
        srcObjectDescriptor.set.call(this, value);
      },
    });
  }

  const originalPlay = HTMLMediaElement.prototype.play;
  HTMLMediaElement.prototype.play = function patchedPlay(...args) {
    return originalPlay.apply(this, args).catch((error) => {
      // NVIDIA's WebRTC component replaces srcObject once during startup.
      // Chrome 150 rejects the first play() with AbortError and the component
      // otherwise tears down an already healthy H.264 session.
      if (error?.name === "AbortError") {
        return undefined;
      }
      throw error;
    });
  };
});

page.on("console", (message) => {
  console.log(`[browser:${message.type()}] ${message.text()}`);
});
page.on("pageerror", (error) => {
  console.error(`[browser:pageerror] ${error.stack ?? error.message}`);
});

if (staticPageUrl) {
  await page.goto(staticPageUrl, { waitUntil: "load", timeout: 60_000 });
  console.log(`Static visualization opened at ${staticPageUrl}`);
  await new Promise(() => {});
}

await page.goto(viewerUrl, { waitUntil: "domcontentloaded", timeout: 60_000 });
console.log(`Isaac Web Viewer loaded at ${viewerUrl}`);
const genericAppOption = page.locator("#no");
await genericAppOption.waitFor({ state: "attached", timeout: 60_000 });
// Chromium runs without a window manager inside Xvfb. Playwright's normal
// actionability checks can wait forever even though React has mounted the
// controls, so trigger the native clicks directly on the attached elements.
await genericAppOption.evaluate((element) => element.click());
const nextButton = page.getByRole("button", { name: "Next" });
await nextButton.waitFor({ state: "attached", timeout: 60_000 });
await nextButton.evaluate((element) => element.click());
console.log(`Isaac Web Viewer opened at ${viewerUrl}`);
const videoCapabilities = await page.evaluate(() =>
  (RTCRtpReceiver.getCapabilities("video")?.codecs ?? []).map((codec) => ({
    mimeType: codec.mimeType,
    clockRate: codec.clockRate,
    fmtpLine: codec.sdpFmtpLine,
  })),
);
console.log(`WebRTC video codecs ${JSON.stringify(videoCapabilities)}`);

setInterval(async () => {
  try {
    const video = await page.locator("video").first();
    const count = await video.count();
    if (count === 0) {
      console.log("Waiting for the WebRTC video element");
      return;
    }
    const state = await video.evaluate((element) => ({
      readyState: element.readyState,
      paused: element.paused,
      width: element.videoWidth,
      height: element.videoHeight,
      currentTime: element.currentTime,
      tracks: element.srcObject instanceof MediaStream
        ? element.srcObject.getVideoTracks().map((track) => ({
            enabled: track.enabled,
            muted: track.muted,
            readyState: track.readyState,
            settings: track.getSettings(),
          }))
        : [],
    }));
    if (state.paused && state.tracks.length > 0) {
      await video.evaluate(async (element) => {
        element.muted = true;
        try {
          await element.play();
        } catch {
          // The next periodic check retries after any source renegotiation.
        }
      });
    }
    console.log(`WebRTC video state ${JSON.stringify(state)}`);
  } catch (error) {
    console.error(`Unable to inspect WebRTC video: ${error.message}`);
  }
}, 10_000);

await new Promise(() => {});
