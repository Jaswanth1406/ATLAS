import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import { Pool } from "pg";
import { NextResponse } from "next/server";

const pool = new Pool({
  connectionString: process.env.NEON_DATABASE_URL,
});

export async function GET() {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { rows } = await pool.query(
      `SELECT id, filename, threshold, road_percentage, avg_confidence, 
              inference_time_ms, image_width, image_height, overlay_b64, mask_b64, created_at
       FROM segmentation_results 
       WHERE user_id = $1 
       ORDER BY created_at DESC 
       LIMIT 50`,
      [session.user.id]
    );

    return NextResponse.json({ results: rows });
  } catch (error) {
    console.error("Fetch history error:", error);
    return NextResponse.json(
      { error: "Failed to fetch history" },
      { status: 500 }
    );
  }
}
