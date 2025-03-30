import express from "express";
import axios from "axios";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const router = express.Router();
router.use(cors());
router.use(express.json());

const GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token";
const GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke";
const { GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET } = process.env;

// Exchange authorization code for tokens
router.post("/exchange-token", async (req, res) => {
  const { code, redirectUri, codeVerifier } = req.body;
  if (!code || !redirectUri || !codeVerifier) {
    return res.status(400).json({ error: "Missing required parameters" });
  }
  try {
    const params = new URLSearchParams({
      code,
      client_id: GOOGLE_CLIENT_ID,
      client_secret: GOOGLE_CLIENT_SECRET,
      redirect_uri: redirectUri,
      grant_type: "authorization_code",
      code_verifier: codeVerifier,
    });

    const { data } = await axios.post(GOOGLE_TOKEN_URL, params.toString(), {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });

    res.json(data);
  } catch (error) {
    console.error("Token exchange error:", error.response ? error.response.data : error.message);
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Internal server error",
    });
  }
});

// Revoke token
router.post("/revoke-token", async (req, res) => {
  const { token } = req.body;
  if (!token) {
    return res.status(400).json({ error: "Missing token" });
  }
  try {
    const params = new URLSearchParams({
      token,
      client_id: GOOGLE_CLIENT_ID,
      client_secret: GOOGLE_CLIENT_SECRET,
    });

    await axios.post(GOOGLE_REVOKE_URL, params.toString(), {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });

    res.json({ success: true });
  } catch (error) {
    console.error("Token revocation error:", error.response ? error.response.data : error.message);
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Internal server error",
    });
  }
});

// Refresh token
router.post("/refresh-token", async (req, res) => {
  const { refreshToken } = req.body;
  if (!refreshToken) {
    return res.status(400).json({ error: "Missing refresh token" });
  }
  try {
    const params = new URLSearchParams({
      refresh_token: refreshToken,
      client_id: GOOGLE_CLIENT_ID,
      client_secret: GOOGLE_CLIENT_SECRET,
      grant_type: "refresh_token",
    });

    const { data } = await axios.post(GOOGLE_TOKEN_URL, params.toString(), {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });

    res.json(data);
  } catch (error) {
    console.error("Token refresh error:", error.response ? error.response.data : error.message);
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Internal server error",
    });
  }
});

export default router;