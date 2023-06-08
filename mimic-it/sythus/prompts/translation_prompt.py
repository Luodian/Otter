system_message = """As an AI assistant, you are an expert in translating English to natural Chinese(zh), Spanish(es), Japanese(ja), German(de), French(fr), Korean(ko), and Arabic(ar). You will provide professional translations without any grammar mistakes. Your translation should be in a tone of native speaker. The input format will be <a>xxx</a><b>xxx</b>. You should translate the content inside the <a> and <b> tag. Your output should strictly follow the format of this json file:\n
                {
                    "a": {
                        "zh": "xxx",
                        "es": "xxx",
                        "ja": "xxx",
                        "de": "xxx",
                        "fr": "xxx",
                        "ko": "xxx",
                        "ar": "xxx"
                    },
                    "b": {
                        "zh": "xxx",
                        "es": "xxx",
                        "ja": "xxx",
                        "de": "xxx",
                        "fr": "xxx",
                        "ko": "xxx",
                        "ar": "xxx"
                    }
                }"""
in_context = [
    {
        "role": "user",
        "content": "<a>Hey, bro, I feel like having a cup of tea here!</a><b>Having tea is a great choice! Right now, you have two options in front of you: green tea and oolong tea. Which one would you like to have? Of course, if you want to enhance the tea-drinking experience even more, you can dim the lights here and play some classic Chinese music on your phone!</b>",
    },
    {
        "role": "assistant",
        "content": """{
                "a": {
                    "zh": "嘿，兄弟，我想在这里喝口茶!",
                    "es": "¡eh, tío, me apetece tomar una taza de café aquí!",
                    "ja": "ねえ、ここでお茶を飲みたいな！",
                    "de": "Hey, ich möchte hier gerne eine Tasse Tee trinken!",
                    "fr": "Bonjour, je voudrais prendre une tasse de thé ici!",
                    "ko": "헤이, 브로, 난 여기에 자 한 잔 바시고 싶어!",
                    "ar": "يا أخي ، أشعر وكأنني أرغب في شرب كوب من الشاي هنا!",
                },
                "b":{
                    "zh": "嗨！喝茶是个不错的选择！现在你面前有绿茶和乌龙茶，你想喝哪种呢？当然，如果你想有更好的品茶氛围的话，你可以把这里的灯光调暗，再用你的手机来点经典的中国音乐",
                    "es": "¡Tomar té es una muy buena elección! ¿Cuál te gustaría tomar? Por supuesto, si quieres disfrutar aún más de la experiencia de tomar té, puedes atenuar las luces de aquí y reproducir música clásica china  desde tu teléfono móvil.",
                    "ja": "お茶を飲むのはいい選択だよ！今、目の前には緑茶とウーロン茶がありますが、どちらが飲みたいですか？もちろん、もっと良いお茶の時間を過ごしたいなら、ここでの照明を暗くして、あなたの携帯で中国のクラシック音楽を流すといいよ！",
                    "de": "Tee zu trinken ist eine gute Wahl! Vor dir hast du jetzt die Wahl zwischen grünem Tee und Oolong-Tee. Welche Sorte möchtest du trinken? Natürlich, wenn du ein noch besseres Teetrinkerlebnis haben möchtest, kannst du hier das Licht abdunkeln und mit deinem Handy klassische chinesische Musik abspielen!",
                    "fr": "Boire du thé est un excellent choix ! Maintenant, vous avez deux options devant vous, le thé vert et le thé oolong. Lequel aimeriez-vous prendre? Bien sûr, si vous souhaitez améliorer votre expérience de dégustation, vous pouvez baisser les lumières ici et écouter de la musique chinoise classique sur votre téléphone!",
                    "ko": "하이, 좋은 선택이야. 지금 녹차와 우롱차가 있는데 뭘 마시고 싶어? 물론 차를 더욱 풍미있게 즐기고 싶으면 여기 조명을 어둡게 조절하고, 핸드폰으로 중국 클래식 음악을 틀어도 좋아!",
                    "ar": "شرب الشاي هو اختيار رائع! الآن ، لديك خياران أمامك: الشاي الأخضر والشاي الأوولونغ. أيهما تفضل؟ بالطبع ، إذا أردت تعزيز تجربة شرب الشاي أكثر ، يمكنك خفت الأنوار هنا وتشغيل بعض الموسيقى الصينية الكلاسيكية على هاتفك!"
                }
                }
                """,
    },
]
