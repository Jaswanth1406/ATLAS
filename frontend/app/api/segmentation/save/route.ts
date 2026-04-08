import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import { Pool } from "pg";
import { NextRequest, NextResponse } from "next/server";

const pool = new Pool({
  connectionString: process.env.NEON_DATABASE_URL,
});

export async function POST(req: NextRequest) {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const {
      filename,
      threshold,
      road_percentage,
      avg_confidence,
      inference_time_ms,
      image_width,
      image_height,
      overlay_b64,
      mask_b64,
    } = body;

    await pool.query(
      `INSERT INTO segmentation_results 
       (user_id, filename, threshold, road_percentage, avg_confidence, inference_time_ms, image_width, image_height, overlay_b64, mask_b64)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)`,
      [
        session.user.id,
        filename,
        threshold,
        road_percentage,
        avg_confidence,
        inference_time_ms,
        image_width,
        image_height,
        overlay_b64,
        mask_b64,
      ]
    );

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Save segmentation error:", error);
    return NextResponse.json(
      { error: "Failed to save result" },
      { status: 500 }
    );
  }
}
