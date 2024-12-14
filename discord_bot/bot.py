from discord.ext import commands
import discord
import os
import boto3
import openai
import qdrant_client
from discord_bot.RAG import generate_graph, add_docs

intents = discord.Intents.default()
intents.message_content = True

openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

q_client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

db = boto3.client("dynamodb", region_name=os.getenv("AWS_REGION"))

bot = commands.Bot(command_prefix="$", intents=intents)
app = generate_graph(openai_client, q_client, db)


@bot.command()
async def q(ctx, *, question: str):
    answer = app.invoke({"question": question})["generation"]
    await ctx.reply(answer)


@bot.command()
async def record(ctx, *, message: str):
    add_docs([message], openai_client, q_client)
    await ctx.reply("Recorded")


bot.run(os.getenv("DISCORD_TOKEN"))
