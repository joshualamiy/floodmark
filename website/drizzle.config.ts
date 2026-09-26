import type { Config } from "drizzle-kit";
import { env } from "$env/static/private";

export default {
	schema: "./src/lib/server/schema.ts",
	out: "./drizzle",
	dialect: "postgresql",
	dbCredentials: {
		url: env.DATABASE_URL!
	},
} satisfies Config;
