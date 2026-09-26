import { env } from "$env/dynamic/private";
import { neon } from "@neondatabase/serverless";
import { drizzle } from "drizzle-orm/neon-http";

type Database = ReturnType<typeof drizzle>;

let database: Database | undefined;

export function getDB(): Database {
	if (!database) {
		const sql = neon(env.DATABASE_URL);
		database = drizzle({ client: sql });
	}

	return database;
}
