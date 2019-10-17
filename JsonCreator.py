import json
import glob
import pprint
from moviepy.editor import VideoFileClip as vc
class Media():
    def __init__(self, id, file_name, type, duration):
        self.duration = duration
        self.id = id
        self.file_name = file_name
        self.type = type


class CommandType:
    def __init__(self, id, event_name):
        self.id = id
        self.event_name = event_name


class Command:
    def __init__(self, id, name, type, trigger_event_id, attached_character_class,
                 relation_class, command_type_id, media_id, duration):
        self.id = id
        self.name = name
        self.type = type
        self.trigger_event_id = trigger_event_id
        self.attached_character_class = attached_character_class
        self.relation_class = relation_class
        self.command_type_id = command_type_id
        self.media_id = media_id
        self.duration = duration


class Data:
    def __init__(self, type):
        self.data = {}
        self.type = type
        self.id = 1
        self.add_data_type = {
            "Media": self.add_media,
            "Command": self.add_command,
            "CommandType": self.add_command_type
        }
        self.create_data_type = {
            "Media": self.create_media_data,
            "Command": self.create_command_data,
            "CommandType": self.create_command_type_data
        }

    def inc_id(self):
        self.id += 1

    def add_item(self, item):
        self.add_data_type[self.type](item)
        self.inc_id()

    def add_media(self, item:Media):

        print('type is', type(item))
        pprint.pprint(item.__dict__)
        print(self.data)
        self.data[self.id] = {
            "id": item.id,
            "file_name": item.file_name,
            "type": item.type,
            "duration": item.duration
        }
        print("end\n")


    def add_command(self, item:Command):
        self.data[self.id] = {
            "id": item.id,
            "name": item.name,
            "type": item.type,
            "trigger_event_id": item.trigger_event_id,
            "attached_character_class": item.attached_character_class,
            "relation_class": item.relation_class,
            "media_id": item.media_id,
            "duration": item.duration
        }

    def add_command_type(self, item:CommandType):
        self.data[self.id] = {
            "id": item.id,
            "event_name": item.event_name
        }



    def create_media_data(self):
        image_names = glob.glob('ImageMem/*')
        video_names = glob.glob('VideoMem/*')

        for img_name in image_names:
            item = Media(self.id, img_name, "I", 0)
            self.add_item(item)

        for avi_name in video_names:
            self.add_item(Media(self.id, avi_name, "V", vc(avi_name).duration*1000))

    def create_command_type_data(self):
        self.add_item(CommandType(self.id, 'Object on the screen'))

    def create_command_data(self, name_person="Andro News", name_class="apple"):
        self.add_item(Command(self.id, name_person, "TB", self.id, name_person, name_class,1, 7, 0 ))
        self.add_item(Command(self.id, "Apple Watch", "TB",  self.id, name_person, name_class, 1, 4, 0))
        self.add_item(Command(self.id,"iPhone 11 Pro Max", "TB", self.id, name_person, name_class, 1, 14, 0))


class JsonCreator:

    def __init__(self, json_name):
        self.media_list = None
        self.json_name = json_name

    def write_in_json(self, data, flag="w" ):
        with open(self.json_name, flag,  encoding='utf-8') as j_file:
            json.dump(data, j_file)
            j_file.close()



def main():
    curr_id = 1
    data = {}

    json_name = "1.json"
    js = JsonCreator(json_name)
    media = Data("Media")
    # command = Data("Command")
    # command_type = Data("CommandType")

    media.create_media_data()
    # command_type.create_command_type_data()
    # command.create_command_data()



    data["media"] = media.data
    # data["command_type"] = command_type.data
    # data["command"] = command.data



    js.write_in_json(data)

if __name__ == '__main__':
    main()

